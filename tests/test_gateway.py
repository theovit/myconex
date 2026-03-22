"""Tests for MYCONEX mesh gateway API."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.gateway.mesh_gateway import MeshGateway, ConnectionManager


class TestConnectionManager(unittest.TestCase):
    """Test WebSocket connection manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConnectionManager()

    def test_connect_disconnect(self):
        """Test connecting and disconnecting WebSocket clients."""
        mock_ws = AsyncMock()

        async def test():
            await self.manager.connect(mock_ws)
            self.assertIn(mock_ws, self.manager.active_connections)

            await self.manager.disconnect(mock_ws)
            self.assertNotIn(mock_ws, self.manager.active_connections)

        asyncio.run(test())

    def test_broadcast(self):
        """Test broadcasting messages to connected clients."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()  # This one will fail

        mock_ws3.send_json.side_effect = Exception("Connection lost")

        async def test():
            await self.manager.connect(mock_ws1)
            await self.manager.connect(mock_ws2)
            await self.manager.connect(mock_ws3)

            message = {"type": "test", "data": "hello"}

            await self.manager.broadcast(message)

            # Verify successful sends
            mock_ws1.send_json.assert_called_once_with(message)
            mock_ws2.send_json.assert_called_once_with(message)

            # Verify failed connection was cleaned up
            self.assertNotIn(mock_ws3, self.manager.active_connections)

        asyncio.run(test())


class TestMeshGateway(unittest.TestCase):
    """Test mesh gateway functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.gateway = MeshGateway(
            nats_url="nats://localhost:4222",
            host="127.0.0.1",
            port=8765
        )

    @patch('core.gateway.mesh_gateway.MeshNATSClient')
    @patch('core.gateway.mesh_gateway.MeshDiscovery')
    def test_startup_success(self, mock_discovery, mock_nats):
        """Test successful gateway startup."""
        mock_nats_instance = AsyncMock()
        mock_nats_instance.is_connected = True
        mock_nats.return_value = mock_nats_instance

        mock_discovery_instance = AsyncMock()
        mock_discovery_instance._running = True
        mock_discovery.return_value = mock_discovery_instance

        async def test():
            await self.gateway._startup()

            self.assertIsNotNone(self.gateway.nats_client)
            self.assertIsNotNone(self.gateway.discovery)
            self.assertTrue(self.gateway._running)

            mock_nats.assert_called_once()
            mock_discovery.assert_called_once()

        asyncio.run(test())

    @patch('core.gateway.mesh_gateway.MeshNATSClient')
    def test_startup_nats_failure(self, mock_nats):
        """Test gateway startup when NATS fails."""
        mock_nats.side_effect = Exception("Connection failed")

        async def test():
            await self.gateway._startup()

            # Should continue without NATS
            self.assertIsNone(self.gateway.nats_client)
            self.assertTrue(self.gateway._running)

        asyncio.run(test())

    def test_create_app(self):
        """Test FastAPI app creation."""
        app = self.gateway._create_app()

        self.assertIsNotNone(app)
        self.assertEqual(app.title, "MYCONEX Mesh Gateway")

        # Check routes exist
        routes = [route.path for route in app.routes]
        self.assertIn("/health", routes)
        self.assertIn("/peers", routes)
        self.assertIn("/task", routes)
        self.assertIn("/chat", routes)
        self.assertIn("/ws", routes)

    @patch('core.gateway.mesh_gateway.MeshNATSClient')
    def test_task_submission(self, mock_nats):
        """Test task submission via REST API."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.submit_task = AsyncMock()
        mock_nats.return_value = mock_client

        self.gateway.nats_client = mock_client

        async def test():
            from fastapi.testclient import TestClient
            from fastapi import FastAPI

            # Create test app
            app = self.gateway._create_app()

            # Override the gateway's nats_client for testing
            self.gateway.nats_client = mock_client

            with TestClient(app) as client:
                response = client.post(
                    "/task",
                    json={
                        "type": "inference",
                        "payload": {"text": "Hello world"},
                        "tier": "T2"
                    }
                )

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("task_id", data)
                self.assertEqual(data["status"], "submitted")

                # Verify NATS was called
                mock_client.submit_task.assert_called_once()

        asyncio.run(test())

    @patch('core.gateway.mesh_gateway.MeshNATSClient')
    def test_chat_endpoint(self, mock_nats):
        """Test chat endpoint."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send_to_tier = AsyncMock()
        mock_nats.return_value = mock_client

        self.gateway.nats_client = mock_client

        async def test():
            from fastapi.testclient import TestClient

            app = self.gateway._create_app()
            self.gateway.nats_client = mock_client

            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={
                        "message": "Hello mesh!",
                        "model": "llama3.2:3b"
                    }
                )

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "sent")

                # Verify message was sent to T2 tier
                mock_client.send_to_tier.assert_called_once()
                args = mock_client.send_to_tier.call_args[0]
                self.assertEqual(args[0], "T2")  # tier

        asyncio.run(test())

    def test_health_endpoint(self):
        """Test health check endpoint."""
        async def test():
            from fastapi.testclient import TestClient

            app = self.gateway._create_app()

            with TestClient(app) as client:
                response = client.get("/health")

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "healthy")
                self.assertIn("services", data)

        asyncio.run(test())

    @patch('core.gateway.mesh_gateway.MeshDiscovery')
    def test_peers_endpoint(self, mock_discovery):
        """Test peers endpoint."""
        mock_discovery_instance = AsyncMock()
        mock_discovery.return_value = mock_discovery_instance
        mock_discovery_instance.registry.get_online = AsyncMock(return_value=[])
        self.gateway.discovery = mock_discovery_instance

        async def test():
            from fastapi.testclient import TestClient

            app = self.gateway._create_app()
            self.gateway.discovery = mock_discovery_instance

            with TestClient(app) as client:
                response = client.get("/peers")

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("peers", data)
                self.assertEqual(data["peers"], [])

        asyncio.run(test())

    def test_task_submission_without_nats(self):
        """Test task submission when NATS is unavailable."""
        self.gateway.nats_client = None

        async def test():
            from fastapi.testclient import TestClient

            app = self.gateway._create_app()

            with TestClient(app) as client:
                response = client.post(
                    "/task",
                    json={
                        "type": "inference",
                        "payload": {"text": "Hello world"}
                    }
                )

                self.assertEqual(response.status_code, 503)
                data = response.json()
                self.assertIn("NATS not connected", data["detail"])

        asyncio.run(test())


if __name__ == '__main__':
    unittest.main()