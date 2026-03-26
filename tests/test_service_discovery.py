"""Tests for MYCONEX service discovery (mDNS hub service auto-discovery)."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.discovery.mesh_discovery import ServiceURLs, ServiceDiscoveryError, MeshDiscovery, ServiceAdvertiser


class TestServiceURLs(unittest.TestCase):

    def test_defaults_are_none(self):
        urls = ServiceURLs()
        self.assertIsNone(urls.nats_url)
        self.assertIsNone(urls.redis_url)
        self.assertIsNone(urls.qdrant_url)

    def test_partial_population(self):
        urls = ServiceURLs(nats_url="nats://192.168.1.10:4222")
        self.assertEqual(urls.nats_url, "nats://192.168.1.10:4222")
        self.assertIsNone(urls.redis_url)

    def test_service_discovery_error_is_runtime_error(self):
        err = ServiceDiscoveryError("NATS not found")
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("NATS", str(err))


class TestMeshDiscoveryZeroconfProperty(unittest.TestCase):

    def test_zeroconf_raises_before_start(self):
        discovery = MeshDiscovery(
            node_name="test-node", tier="T3", roles=["relay"]
        )
        with self.assertRaises(RuntimeError) as ctx:
            _ = discovery.zeroconf
        self.assertIn("not started", str(ctx.exception))

    @patch('core.discovery.mesh_discovery.AsyncZeroconf')
    @patch('core.discovery.mesh_discovery.AsyncServiceBrowser')
    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_zeroconf_returns_instance_after_start(
        self, mock_info, mock_browser, mock_zc
    ):
        mock_zc_instance = AsyncMock()
        mock_zc.return_value = mock_zc_instance

        discovery = MeshDiscovery(
            node_name="test-node", tier="T3", roles=["relay"]
        )

        async def run():
            await discovery.start()
            zc = discovery.zeroconf
            self.assertIs(zc, mock_zc_instance)
            await discovery.stop()

        asyncio.run(run())


class TestServiceAdvertiser(unittest.TestCase):

    def _make_advertiser(self, mock_zc):
        """Build a ServiceAdvertiser with a fake config and zeroconf."""
        cfg = MagicMock()
        cfg.mesh.nats_url = "nats://localhost:4222"
        cfg.mesh.redis_url = "redis://localhost:6379"
        cfg.mesh.qdrant_url = "http://localhost:6333"
        return ServiceAdvertiser(zc=mock_zc, node_name="hub-node", cfg=cfg)

    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_registers_services_for_open_ports(self, mock_service_info):
        """Registers mDNS records only for ports that respond."""
        mock_zc = AsyncMock()
        mock_zc.async_register_service = AsyncMock()
        mock_si = AsyncMock()
        mock_service_info.return_value = mock_si

        advertiser = self._make_advertiser(mock_zc)

        async def run():
            # Patch _probe_port so NATS and Qdrant respond, Redis does not
            async def fake_probe(port, timeout=1.0):
                return port in (4222, 6333)

            advertiser._probe_port = fake_probe
            await advertiser.start()
            # Should register two services
            self.assertEqual(mock_zc.async_register_service.call_count, 2)

        asyncio.run(run())

    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_silent_on_all_probe_failures(self, mock_service_info):
        """No error raised if no services respond (normal for worker nodes)."""
        mock_zc = AsyncMock()
        mock_zc.async_register_service = AsyncMock()
        advertiser = self._make_advertiser(mock_zc)

        async def run():
            async def always_false(port, timeout=1.0):
                return False

            advertiser._probe_port = always_false
            await advertiser.start()  # must not raise
            mock_zc.async_register_service.assert_not_called()

        asyncio.run(run())

    def test_probe_port_returns_false_on_connection_refused(self):
        """_probe_port returns False when nothing is listening."""
        mock_zc = AsyncMock()
        cfg = MagicMock()
        cfg.mesh.nats_url = "nats://localhost:4222"
        cfg.mesh.redis_url = "redis://localhost:6379"
        cfg.mesh.qdrant_url = "http://localhost:6333"
        advertiser = ServiceAdvertiser(zc=mock_zc, node_name="test", cfg=cfg)

        async def run():
            # Port 1 will always be refused
            result = await advertiser._probe_port(1, timeout=0.1)
            self.assertFalse(result)

        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
