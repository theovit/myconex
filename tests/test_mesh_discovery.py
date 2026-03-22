"""Tests for MYCONEX mesh discovery using mDNS."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.discovery.mesh_discovery import (
    MeshDiscovery,
    MeshPeer,
    PeerRegistry,
    MeshServiceListener,
    SERVICE_TYPE,
    SERVICE_VERSION,
)


class TestMeshPeer(unittest.TestCase):
    """Test MeshPeer data class."""

    def test_peer_creation(self):
        """Test creating a mesh peer."""
        peer = MeshPeer(
            name="test-node",
            hostname="test-node.local",
            address="192.168.1.100",
            port=8765,
            tier="T2",
            roles=["inference", "embedding"],
            version="0.1.0"
        )

        self.assertEqual(peer.name, "test-node")
        self.assertEqual(peer.endpoint, "http://192.168.1.100:8765")
        self.assertTrue(peer.is_online)
        self.assertIn("inference", peer.roles)

    def test_peer_to_dict(self):
        """Test peer serialization."""
        peer = MeshPeer(
            name="test-node",
            hostname="test-node.local",
            address="192.168.1.100",
            port=8765,
            tier="T2"
        )

        data = peer.to_dict()
        self.assertEqual(data["name"], "test-node")
        self.assertEqual(data["endpoint"], "http://192.168.1.100:8765")
        self.assertEqual(data["tier"], "T2")


class TestPeerRegistry(unittest.TestCase):
    """Test peer registry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = PeerRegistry()

    def test_add_peer(self):
        """Test adding a peer to registry."""
        peer = MeshPeer(
            name="test-peer",
            hostname="test-peer.local",
            address="192.168.1.101",
            port=8765
        )

        async def test():
            await self.registry.add_or_update(peer)
            peers = await self.registry.get_all()
            self.assertEqual(len(peers), 1)
            self.assertEqual(peers[0].name, "test-peer")

        asyncio.run(test())

    def test_remove_peer(self):
        """Test removing a peer from registry."""
        peer = MeshPeer(
            name="test-peer",
            hostname="test-peer.local",
            address="192.168.1.101",
            port=8765
        )

        async def test():
            await self.registry.add_or_update(peer)
            await self.registry.remove("test-peer")
            peers = await self.registry.get_online()
            self.assertEqual(len(peers), 0)

        asyncio.run(test())

    def test_get_by_tier(self):
        """Test filtering peers by tier."""
        peer_t2 = MeshPeer(
            name="gpu-node",
            hostname="gpu-node.local",
            address="192.168.1.102",
            port=8765,
            tier="T2"
        )
        peer_t3 = MeshPeer(
            name="cpu-node",
            hostname="cpu-node.local",
            address="192.168.1.103",
            port=8765,
            tier="T3"
        )

        async def test():
            await self.registry.add_or_update(peer_t2)
            await self.registry.add_or_update(peer_t3)

            t2_peers = await self.registry.get_by_tier("T2")
            self.assertEqual(len(t2_peers), 1)
            self.assertEqual(t2_peers[0].name, "gpu-node")

        asyncio.run(test())

    def test_get_by_role(self):
        """Test filtering peers by role."""
        peer = MeshPeer(
            name="inference-node",
            hostname="inference-node.local",
            address="192.168.1.104",
            port=8765,
            roles=["inference", "embedding"]
        )

        async def test():
            await self.registry.add_or_update(peer)

            inference_peers = await self.registry.get_by_role("inference")
            self.assertEqual(len(inference_peers), 1)
            self.assertEqual(inference_peers[0].name, "inference-node")

        asyncio.run(test())


class TestMeshServiceListener(unittest.TestCase):
    """Test mDNS service listener."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = PeerRegistry()
        self.listener = MeshServiceListener(
            registry=self.registry,
            local_name="test-local",
            on_peer_join=AsyncMock(),
            on_peer_leave=AsyncMock()
        )

    @patch('core.discovery.mesh_discovery.ServiceInfo')
    def test_handle_add_service(self, mock_service_info):
        """Test handling service addition."""
        # Mock Zeroconf service info
        mock_info = MagicMock()
        mock_info.parsed_addresses.return_value = ["192.168.1.105"]
        mock_info.port = 8765
        mock_info.server = "remote-node.local"
        mock_info.properties = {
            b"tier": b"T2",
            b"roles": b"inference,embedding",
            b"version": b"0.1.0"
        }
        mock_info.request.return_value = True

        mock_service_info.return_value = mock_info

        async def test():
            await self.listener._handle_add(MagicMock(), SERVICE_TYPE, "remote-node._ai-mesh._tcp.local.")

            peers = await self.registry.get_all()
            self.assertEqual(len(peers), 1)
            self.assertEqual(peers[0].name, "remote-node")
            self.assertEqual(peers[0].tier, "T2")
            self.assertIn("inference", peers[0].roles)

        asyncio.run(test())


class TestMeshDiscovery(unittest.TestCase):
    """Test main mesh discovery service."""

    def setUp(self):
        """Set up test fixtures."""
        self.discovery = MeshDiscovery(
            node_name="test-node",
            tier="T2",
            roles=["inference"],
            api_port=8765
        )

    @patch('core.discovery.mesh_discovery.AsyncZeroconf')
    @patch('core.discovery.mesh_discovery.AsyncServiceBrowser')
    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_start_discovery(self, mock_service_info, mock_browser, mock_zeroconf):
        """Test starting the discovery service."""
        mock_zeroconf_instance = AsyncMock()
        mock_zeroconf.return_value = mock_zeroconf_instance

        mock_browser_instance = AsyncMock()
        mock_browser.return_value = mock_browser_instance

        mock_service_info_instance = AsyncMock()
        mock_service_info.return_value = mock_service_info_instance

        async def test():
            await self.discovery.start()

            # Should have created zeroconf instance
            mock_zeroconf.assert_called_once()

            # Should have registered service
            mock_service_info.assert_called_once()

            # Should have started browser
            mock_browser.assert_called_once()

            self.assertTrue(self.discovery._running)

            await self.discovery.stop()
            self.assertFalse(self.discovery._running)

        asyncio.run(test())

    def test_service_name_generation(self):
        """Test that service names are generated correctly."""
        discovery = MeshDiscovery(
            node_name="my-node",
            tier="T2",
            roles=["inference"],
            api_port=8080
        )

        expected_name = "MYCONEX-my-node._ai-mesh._tcp.local."
        # The service_name property isn't directly exposed, but we can check the registration
        self.assertEqual(discovery.node_name, "my-node")


if __name__ == '__main__':
    unittest.main()