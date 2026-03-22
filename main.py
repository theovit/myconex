"""
MYCONEX — Distributed AI Mesh System
Entry point: starts the mesh node, coordinator, task router, and optional API.

Usage:
    python main.py                          # Auto-detect config
    python main.py --config /etc/myconex/node.yaml
    python main.py --mode api               # Also start REST API
    python main.py --mode worker            # Worker-only (no API)
    python main.py status                   # Print node status and exit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("myconex")
console = Console()

# ─── Version ──────────────────────────────────────────────────────────────────
VERSION = "0.1.0"
BANNER = f"""[cyan bold]
 ███╗   ███╗██╗   ██╗ ██████╗ ██████╗ ███╗   ██╗███████╗██╗  ██╗
 ████╗ ████║╚██╗ ██╔╝██╔════╝██╔═══██╗████╗  ██║██╔════╝╚██╗██╔╝
 ██╔████╔██║ ╚████╔╝ ██║     ██║   ██║██╔██╗ ██║█████╗   ╚███╔╝
 ██║╚██╔╝██║  ╚██╔╝  ██║     ██║   ██║██║╚██╗██║██╔══╝   ██╔██╗
 ██║ ╚═╝ ██║   ██║   ╚██████╗╚██████╔╝██║ ╚████║███████╗██╔╝ ██╗
 ╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
[/cyan bold][dim] Distributed AI Mesh System v{VERSION} — Inspired by fungal networks[/dim]"""


# ─── Config Loading ───────────────────────────────────────────────────────────

def load_config(config_path: Optional[str]) -> dict:
    """Load and merge mesh_config.yaml with optional node override."""
    base_dir = Path(__file__).parent
    default_config_path = base_dir / "config" / "mesh_config.yaml"

    config = {}

    # Load base config
    if default_config_path.exists():
        with open(default_config_path) as f:
            config = yaml.safe_load(f) or {}

    # Load node-specific override
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            node_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, node_config)
    elif config_path:
        logger.warning(f"Config file not found: {config_path}, using defaults.")

    # Apply env overrides
    _apply_env_overrides(config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_env_overrides(config: dict) -> None:
    env_map = {
        "NATS_URL":       ("nats", "url"),
        "REDIS_URL":      ("redis", "url"),
        "QDRANT_URL":     ("qdrant", "url"),
        "OLLAMA_URL":     ("ollama", "url"),
        "LITELLM_URL":    ("litellm", "url"),
        "MYCONEX_NODE":   ("node", "name"),
        "MYCONEX_TIER":   ("node", "tier"),
    }
    for env_key, (section, key) in env_map.items():
        val = os.environ.get(env_key)
        if val:
            config.setdefault(section, {})[key] = val


# ─── Node Bootstrap ───────────────────────────────────────────────────────────

async def run_node(config: dict, mode: str) -> None:
    from core.coordinator.orchestrator import MeshOrchestrator
    from orchestration.workflows.task_router import TaskRouter

    # Load .env for secrets (DISCORD_BOT_TOKEN etc.) — no-op if already loaded
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass

    # 1. Start orchestrator (handles hardware detection, mDNS, NATS)
    orchestrator = MeshOrchestrator(config)
    await orchestrator.start()

    # 2. Start task router
    router = TaskRouter(
        node_tier=orchestrator._hardware.tier,
        ollama_url=config.get("ollama", {}).get("url", "http://localhost:11434"),
        litellm_url=config.get("litellm", {}).get("url", "http://localhost:4000"),
    )
    router.set_remote_handler(orchestrator)
    await router.start()

    # 3. Print startup status
    _print_status(orchestrator, router)

    # 4. Optionally start API server
    api_server = None
    if mode in ("api", "full"):
        api_server = asyncio.create_task(
            run_api(config, orchestrator, router)
        )

    # 5. Optionally start Discord gateway
    discord_task = None
    discord_cfg = config.get("discord", {})
    if discord_cfg.get("enabled") or mode == "discord":
        try:
            from core.gateway.discord_gateway import DiscordGateway
            discord_gw = DiscordGateway(config, router)
            discord_task = asyncio.create_task(discord_gw.start())
            console.print("[cyan]Discord gateway starting...[/cyan]")
        except Exception as e:
            logger.error("Failed to start Discord gateway: %s", e)

    # 6. Run until interrupted
    stop_event = asyncio.Event()

    def _handle_signal(sig):
        logger.info(f"Received signal {sig.name}, shutting down...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))

    console.print("[green]Node running. Press Ctrl+C to stop.[/green]\n")
    await stop_event.wait()

    # Shutdown
    console.print("\n[yellow]Shutting down...[/yellow]")
    if discord_task:
        discord_task.cancel()
        try:
            from core.gateway.discord_gateway import DiscordGateway
            await discord_gw.stop()
        except Exception:
            pass
    if api_server:
        api_server.cancel()
    await router.stop()
    await orchestrator.stop()
    console.print("[green]Goodbye.[/green]")


# ─── API Server ───────────────────────────────────────────────────────────────

async def run_api(config: dict, orchestrator, router) -> None:
    """Start the aiohttp API gateway (TaskRouter + session management)."""
    try:
        from core.gateway.api_gateway import APIGateway

        api_cfg = config.get("api", {})
        host = api_cfg.get("host", "127.0.0.1")
        port = int(api_cfg.get("port", 8765))
        console.print(f"[cyan]API gateway listening on http://{host}:{port}[/cyan]")

        gw = APIGateway(config, router)
        await gw.start()

    except ImportError as e:
        logger.error("aiohttp not installed. Cannot start API gateway. Run: pip install aiohttp — %s", e)
    except Exception as e:
        logger.error("API gateway error: %s", e, exc_info=True)


# ─── Status Display ───────────────────────────────────────────────────────────

def _print_status(orchestrator, router) -> None:
    hw = orchestrator._hardware
    st = orchestrator.status()

    console.print(BANNER)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value", style="bold")

    table.add_row("Node",    st["node"])
    table.add_row("Tier",    f"{st['tier']} — {hw.tier_label}")
    table.add_row("Roles",   ", ".join(st["roles"]))
    table.add_row("CPU",     f"{hw.cpu_model} ({hw.cpu_cores_logical} cores)")
    table.add_row("RAM",     f"{hw.ram_total_gb} GB")
    table.add_row("GPU",     f"{hw.gpu_name} ({hw.gpu_vram_gb} GB VRAM)")
    table.add_row("NATS",    "[green]connected[/green]" if st["nats_connected"] else "[yellow]offline[/yellow]")

    console.print(Panel(table, title="[bold cyan]MYCONEX Node[/bold cyan]", border_style="cyan"))


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--config", "-c", default=None, envvar="MYCONEX_CONFIG", help="Path to node config YAML")
@click.option("--mode", "-m", default="worker", type=click.Choice(["worker", "api", "discord", "full"]), help="Node mode (discord = mesh+bot, full = mesh+api+bot)")
@click.option("--host", default=None, help="API host override")
@click.option("--port", default=None, type=int, help="API port override")
@click.option("--log-level", default="INFO", help="Logging level")
def cli(ctx, config, mode, host, port, log_level):
    """MYCONEX — Distributed AI Mesh Node"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["mode"] = mode
    ctx.obj["host"] = host
    ctx.obj["port"] = port

    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if ctx.invoked_subcommand is None:
        cfg = load_config(config)
        if host:
            cfg.setdefault("api", {})["host"] = host
        if port:
            cfg.setdefault("api", {})["port"] = port

        try:
            asyncio.run(run_node(cfg, mode))
        except KeyboardInterrupt:
            pass


@cli.command()
@click.pass_context
def status(ctx):
    """Print hardware profile and mesh status."""
    from core.classifier.hardware import HardwareDetector
    detector = HardwareDetector()
    profile = detector.detect()

    console.print(BANNER)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value", style="bold")

    table.add_row("Hostname",   profile.hostname)
    table.add_row("OS",         f"{profile.os_name} {profile.os_version}")
    table.add_row("CPU",        f"{profile.cpu_model} ({profile.cpu_cores_logical} cores)")
    table.add_row("RAM",        f"{profile.ram_total_gb} GB")
    table.add_row("GPU",        f"{profile.gpu_name} ({profile.gpu_vram_gb} GB VRAM)")
    table.add_row("Tier",       f"[bold cyan]{profile.tier}[/bold cyan] — {profile.tier_label}")
    table.add_row("Roles",      ", ".join(profile.roles))
    table.add_row("Max Model",  profile.capabilities["max_model_size"])
    table.add_row("Ollama",     profile.capabilities["recommended_ollama_model"])

    console.print(Panel(table, title="[cyan bold]Hardware Profile[/cyan bold]", border_style="cyan"))


@cli.command()
@click.argument("prompt")
@click.option("--model", default=None)
@click.option("--tier", default=None, help="Target tier")
@click.pass_context
def ask(ctx, prompt, model, tier):
    """Send a one-shot prompt to the mesh."""
    async def _ask():
        config_path = ctx.obj.get("config_path")
        cfg = load_config(config_path)

        from orchestration.workflows.task_router import TaskRouter
        from core.classifier.hardware import HardwareDetector

        hw = HardwareDetector().detect()
        router = TaskRouter(
            node_tier=hw.tier,
            ollama_url=cfg.get("ollama", {}).get("url", "http://localhost:11434"),
        )
        await router.start()

        result = await router.route("chat", {"prompt": prompt}, force_local=True)

        if result.success:
            output = result.output or {}
            response = output.get("response", str(output))
            console.print(f"\n[bold]{response}[/bold]\n")
        else:
            console.print(f"[red]Error: {result.error}[/red]")

        await router.stop()

    asyncio.run(_ask())


@cli.command()
@click.pass_context
def hardware(ctx):
    """Run hardware detection and output JSON."""
    from core.classifier.hardware import detect_and_classify
    data = detect_and_classify()
    click.echo(json.dumps(data, indent=2))


if __name__ == "__main__":
    cli()
