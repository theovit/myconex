"""
MYCONEX Hardware Classifier
Detects system hardware and classifies nodes into tiers:
  T1: >24 GB VRAM  — Apex / Heavy GPU
  T2: 8–24 GB VRAM — Mid GPU
  T3: CPU ≥16 cores or ≥16 GB RAM, no significant GPU
  T4: RPi / <8 GB RAM / Edge device
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import psutil


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_mb: int
    vram_gb: float
    driver_version: Optional[str] = None
    vendor: str = "unknown"   # nvidia | amd | intel | apple


@dataclass
class HardwareProfile:
    # CPU
    cpu_model: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    cpu_arch: str
    cpu_freq_mhz: float

    # RAM
    ram_total_gb: float
    ram_available_gb: float

    # GPU (best/primary)
    gpu_name: str
    gpu_vram_gb: float
    gpu_vendor: str
    gpus: list[GPUInfo] = field(default_factory=list)

    # Platform flags
    is_raspberry_pi: bool = False
    rpi_model: Optional[str] = None
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""

    # Disk
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0

    # Classification results
    tier: str = ""
    tier_label: str = ""
    roles: list[str] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)


# ─── Tier Definitions ─────────────────────────────────────────────────────────

TIER_DEFINITIONS = {
    "T1": {
        "label": "Apex — Heavy GPU",
        "roles": ["large-model", "training", "heavy-inference", "coordinator", "embedding"],
        "description": "High-end GPU node with >24 GB VRAM. Can run 70B+ models.",
        "ollama_model": "llama3.1:70b",
    },
    "T2": {
        "label": "Mid-GPU",
        "roles": ["medium-model", "inference", "embedding", "fine-tuning"],
        "description": "Mid-range GPU node with 8–24 GB VRAM. Runs 7B–30B models.",
        "ollama_model": "llama3.1:8b",
    },
    "T3": {
        "label": "CPU-Heavy",
        "roles": ["orchestration", "embedding", "lightweight-inference", "relay"],
        "description": "High-core-count CPU or large RAM without significant GPU.",
        "ollama_model": "llama3.2:3b",
    },
    "T4": {
        "label": "Edge / Embedded",
        "roles": ["sensor", "relay", "lightweight-inference"],
        "description": "Raspberry Pi or low-resource node. Minimal inference only.",
        "ollama_model": "phi3:mini",
    },
}


# ─── Detection Helpers ────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a subprocess and return stdout, empty string on error."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return ""


def _detect_gpus_nvidia() -> list[GPUInfo]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram_mb = int(parts[2])
            driver = parts[3] if len(parts) > 3 else None
            gpus.append(GPUInfo(
                index=idx,
                name=name,
                vram_mb=vram_mb,
                vram_gb=round(vram_mb / 1024, 1),
                driver_version=driver,
                vendor="nvidia",
            ))
        except (ValueError, IndexError):
            continue
    return gpus


def _detect_gpus_amd() -> list[GPUInfo]:
    """Detect AMD GPUs via rocm-smi or lspci fallback."""
    gpus: list[GPUInfo] = []

    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if out:
        for i, line in enumerate(out.splitlines()):
            if "Device" in line or not line.strip():
                continue
            parts = line.split(",")
            try:
                vram_bytes = int(parts[1].strip())
                vram_mb = vram_bytes // (1024 * 1024)
                gpus.append(GPUInfo(
                    index=i,
                    name=f"AMD GPU {i}",
                    vram_mb=vram_mb,
                    vram_gb=round(vram_mb / 1024, 1),
                    vendor="amd",
                ))
            except (ValueError, IndexError):
                continue
    else:
        # lspci fallback — no VRAM info
        lspci = _run(["lspci"])
        for i, line in enumerate(lspci.splitlines()):
            if re.search(r"(VGA|3D|Display).*(AMD|Radeon)", line, re.IGNORECASE):
                gpus.append(GPUInfo(
                    index=i,
                    name=line.split(":")[-1].strip(),
                    vram_mb=0,
                    vram_gb=0.0,
                    vendor="amd",
                ))

    return gpus


def _detect_rpi() -> tuple[bool, Optional[str]]:
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        model = model_path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
        if "raspberry" in model.lower():
            return True, model
    # /sys/firmware fallback
    board_path = Path("/sys/firmware/devicetree/base/model")
    if board_path.exists():
        model = board_path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
        if "raspberry" in model.lower():
            return True, model
    return False, None


# ─── Main Detector ────────────────────────────────────────────────────────────

class HardwareDetector:
    def detect(self) -> HardwareProfile:
        # CPU
        cpu_freq = psutil.cpu_freq()
        cpu_model = self._get_cpu_model()

        # RAM
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024 ** 3), 1)
        ram_avail_gb = round(mem.available / (1024 ** 3), 1)

        # Disk
        disk = psutil.disk_usage("/")
        disk_total_gb = round(disk.total / (1024 ** 3), 1)
        disk_free_gb = round(disk.free / (1024 ** 3), 1)

        # OS
        uname = platform.uname()

        # RPi
        is_rpi, rpi_model = _detect_rpi()

        # GPUs
        gpus = _detect_gpus_nvidia() or _detect_gpus_amd()

        # Primary GPU stats
        primary_gpu = gpus[0] if gpus else None
        gpu_name = primary_gpu.name if primary_gpu else "None"
        gpu_vram_gb = primary_gpu.vram_gb if primary_gpu else 0.0
        gpu_vendor = primary_gpu.vendor if primary_gpu else "none"

        profile = HardwareProfile(
            cpu_model=cpu_model,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_arch=uname.machine,
            cpu_freq_mhz=round(cpu_freq.max if cpu_freq else 0.0, 1),
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_avail_gb,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            gpu_vendor=gpu_vendor,
            gpus=gpus,
            is_raspberry_pi=is_rpi,
            rpi_model=rpi_model,
            os_name=uname.system,
            os_version=uname.release,
            hostname=platform.node(),
            disk_total_gb=disk_total_gb,
            disk_free_gb=disk_free_gb,
        )

        # Classify
        tier, tier_def = self._classify(profile)
        profile.tier = tier
        profile.tier_label = tier_def["label"]
        profile.roles = tier_def["roles"]
        profile.capabilities = self._build_capabilities(profile, tier_def)

        return profile

    def _get_cpu_model(self) -> str:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "Unknown CPU"

    def _classify(self, p: HardwareProfile) -> tuple[str, dict]:
        if p.is_raspberry_pi or p.ram_total_gb < 8:
            return "T4", TIER_DEFINITIONS["T4"]
        if p.gpu_vram_gb >= 24:
            return "T1", TIER_DEFINITIONS["T1"]
        if p.gpu_vram_gb >= 8:
            return "T2", TIER_DEFINITIONS["T2"]
        if p.cpu_cores_logical >= 16 or p.ram_total_gb >= 16:
            return "T3", TIER_DEFINITIONS["T3"]
        return "T4", TIER_DEFINITIONS["T4"]

    def _build_capabilities(self, p: HardwareProfile, tier_def: dict) -> dict:
        return {
            "max_model_size": self._max_model_size(p),
            "can_train": p.gpu_vram_gb >= 24,
            "can_embed": p.ram_total_gb >= 8,
            "can_orchestrate": p.tier in ("T1", "T2", "T3"),
            "supports_docker": not p.is_raspberry_pi or p.ram_total_gb >= 4,
            "recommended_ollama_model": tier_def.get("ollama_model", "phi3:mini"),
            "concurrent_requests": max(1, p.cpu_cores_logical // 4),
        }

    def _max_model_size(self, p: HardwareProfile) -> str:
        vram = p.gpu_vram_gb
        ram = p.ram_total_gb
        if vram >= 48:
            return "70B+"
        if vram >= 24:
            return "70B"
        if vram >= 16:
            return "30B"
        if vram >= 8:
            return "13B"
        if ram >= 32:
            return "7B (CPU)"
        if ram >= 16:
            return "3B (CPU)"
        return "1B (CPU)"


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def detect_and_classify() -> dict:
    detector = HardwareDetector()
    profile = detector.detect()
    result = asdict(profile)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MYCONEX Hardware Classifier")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--tier-only", action="store_true", help="Print tier only")
    args = parser.parse_args()

    data = detect_and_classify()

    if args.tier_only:
        print(data["tier"])
        return

    if args.json:
        print(json.dumps(data, indent=2))
        return

    # Human-readable summary
    print(f"\nMYCONEX Hardware Profile")
    print(f"{'─'*40}")
    print(f"  Hostname  : {data['hostname']}")
    print(f"  OS        : {data['os_name']} {data['os_version']}")
    print(f"  CPU       : {data['cpu_model']}")
    print(f"  Cores     : {data['cpu_cores_physical']}P / {data['cpu_cores_logical']}L @ {data['cpu_freq_mhz']} MHz")
    print(f"  RAM       : {data['ram_total_gb']} GB total, {data['ram_available_gb']} GB free")
    print(f"  GPU       : {data['gpu_name']}")
    print(f"  VRAM      : {data['gpu_vram_gb']} GB")
    print(f"  Disk      : {data['disk_free_gb']} GB free / {data['disk_total_gb']} GB total")
    if data["is_raspberry_pi"]:
        print(f"  RPi Model : {data['rpi_model']}")
    print(f"\n  {'─'*36}")
    print(f"  Tier      : {data['tier']} — {data['tier_label']}")
    print(f"  Roles     : {', '.join(data['roles'])}")
    print(f"  Max Model : {data['capabilities']['max_model_size']}")
    print(f"  Ollama    : {data['capabilities']['recommended_ollama_model']}")
    print()


if __name__ == "__main__":
    main()
