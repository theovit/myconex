"""Tests for MYCONEX hardware detection and classification."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.classifier.hardware import HardwareDetector, HardwareProfile, TIER_DEFINITIONS, _detect_gpus_nvidia


class TestHardwareDetector(unittest.TestCase):
    """Test hardware detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = HardwareDetector()

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_count')
    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_basic_hardware(self, mock_machine, mock_system, mock_cpu_count, mock_memory):
        """Test basic hardware detection."""
        # Mock system info
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        mock_cpu_count.return_value = 8
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16 GB

        profile = self.detector.detect()

        self.assertEqual(profile.os_name, "Linux")
        self.assertEqual(profile.cpu_arch, "x86_64")
        self.assertEqual(profile.cpu_cores_logical, 8)
        self.assertEqual(profile.ram_total_gb, 16.0)

    @patch('core.classifier.hardware._run')
    def test_detect_nvidia_gpu(self, mock_subprocess):
        """Test NVIDIA GPU detection."""
        # Mock nvidia-smi output
        mock_run = mock_subprocess
        mock_run.return_value = "0, GeForce RTX 3060, 8192, 535.129.03"

        detector = HardwareDetector()
        gpus = _detect_gpus_nvidia()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].name, "GeForce RTX 3060")
        self.assertEqual(gpus[0].vram_mb, 8192)

    def test_tier_classification_t4(self):
        """Test T4 (edge) tier classification."""
        profile = HardwareProfile(
            cpu_model="Raspberry Pi",
            cpu_cores_logical=4,
            cpu_cores_physical=4,
            cpu_arch="arm64",
            cpu_freq_mhz=1500.0,
            ram_total_gb=4.0,
            ram_available_gb=3.5,
            gpu_name="",
            gpu_vram_gb=0.0,
            gpu_vendor="none",
            is_raspberry_pi=True
        )

        detector = HardwareDetector()
        tier, tier_def = detector._classify(profile)
        profile.tier = tier
        profile.tier_label = tier_def['label']
        profile.roles = tier_def['roles']
        profile.capabilities = detector._build_capabilities(profile, tier_def)

        self.assertEqual(profile.tier, "T4")
        self.assertEqual(profile.tier_label, "Edge / Embedded")
        self.assertIn("sensor", profile.roles)
        self.assertIn("relay", profile.roles)

    def test_tier_classification_t2(self):
        """Test T2 (mid-GPU) tier classification."""
        profile = HardwareProfile(
            cpu_model="Intel Core i7",
            cpu_cores_logical=16,
            cpu_cores_physical=8,
            cpu_arch="x86_64",
            cpu_freq_mhz=4000.0,
            ram_total_gb=32.0,
            ram_available_gb=28.0,
            gpu_name="NVIDIA GeForce RTX 4060",
            gpu_vram_gb=8.0,
            gpu_vendor="nvidia",
        )

        detector = HardwareDetector()
        tier, tier_def = detector._classify(profile)
        profile.tier = tier
        profile.tier_label = tier_def['label']
        profile.roles = tier_def['roles']
        profile.capabilities = detector._build_capabilities(profile, tier_def)

        self.assertEqual(profile.tier, "T2")
        self.assertEqual(profile.tier_label, "Mid-GPU")
        self.assertIn("inference", profile.roles)
        self.assertIn("embedding", profile.roles)

    def test_tier_classification_t1(self):
        """Test T1 (heavy GPU) tier classification."""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen 9",
            cpu_cores_logical=32,
            cpu_cores_physical=16,
            cpu_arch="x86_64",
            cpu_freq_mhz=4000.0,
            ram_total_gb=64.0,
            ram_available_gb=60.0,
            gpu_name="NVIDIA RTX 4090",
            gpu_vram_gb=24.0,
            gpu_vendor="nvidia",
        )

        detector = HardwareDetector()
        tier, tier_def = detector._classify(profile)
        profile.tier = tier
        profile.tier_label = tier_def['label']
        profile.roles = tier_def['roles']
        profile.capabilities = detector._build_capabilities(profile, tier_def)

        self.assertEqual(profile.tier, "T1")
        self.assertEqual(profile.tier_label, "Apex — Heavy GPU")
        self.assertIn("large-model", profile.roles)
        self.assertIn("training", profile.roles)

    def test_capabilities_assignment(self):
        """Test that capabilities are properly assigned based on tier."""
        profile = HardwareProfile(
            cpu_model="Intel Core i5",
            cpu_cores_logical=8,
            cpu_cores_physical=4,
            cpu_arch="x86_64",
            cpu_freq_mhz=3000.0,
            ram_total_gb=16.0,
            ram_available_gb=14.0,
            gpu_name="",
            gpu_vram_gb=0.0,
            gpu_vendor="none",
        )

        detector = HardwareDetector()
        tier, tier_def = detector._classify(profile)
        profile.tier = tier
        profile.tier_label = tier_def['label']
        profile.roles = tier_def['roles']
        profile.capabilities = detector._build_capabilities(profile, tier_def)

        self.assertEqual(profile.tier, "T3")
        self.assertIn("max_model_size", profile.capabilities)
        self.assertIn("recommended_ollama_model", profile.capabilities)


if __name__ == '__main__':
    unittest.main()