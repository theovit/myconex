"""Tests for MYCONEX NATS messaging client."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.messaging.nats_client import (
    MeshNATSClient,
    MeshMessage,
    HeartbeatService,
    SUBJECT_BROADCAST,
    SUBJECT_HEARTBEAT,
    subject_node,
    subject_tier,
    subject_task_result,
)


class TestMeshMessage(unittest.TestCase):
    """Test MeshMessage data class and serialization."""

    def test_message_creation(self):
        """Test creating a mesh message."""
        msg = MeshMessage(
            subject="mesh.broadcast",
            payload={"hello": "world"},
            sender="test-node"
        )

        self.assertEqual(msg.subject, "mesh.broadcast")
        self.assertEqual(msg.payload, {"hello": "world"})
        self.assertEqual(msg.sender, "test-node")
        self.assertIsNotNone(msg.msg_id)
        self.assertIsNotNone(msg.timestamp)

    def test_message_encode_decode(self):
        """Test message serialization and deserialization."""
        original = MeshMessage(
            subject="mesh.node.target",
            payload={"task": "inference", "data": [1, 2, 3]},
            sender="source-node",
            msg_id="test-123"
        )

        encoded = original.encode()
        decoded = MeshMessage.decode(encoded)

        self.assertEqual(decoded.subject, original.subject)
        self.assertEqual(decoded.payload, original.payload)
        self.assertEqual(decoded.sender, original.sender)
        self.assertEqual(decoded.msg_id, original.msg_id)

    def test_subject_functions(self):
        """Test subject generation functions."""
        self.assertEqual(subject_node("mynode"), "mesh.node.mynode")
        self.assertEqual(subject_tier("T2"), "mesh.tier.T2")
        self.assertEqual(subject_task_result("task-123"), "mesh.task.result.task-123")


class TestMeshNATSClient(unittest.TestCase):
    """Test NATS client functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MeshNATSClient(
            node_name="test-node",
            nats_url="nats://localhost:4222"
        )

    @patch('nats.connect')
    def test_connect_success(self, mock_connect):
        """Test successful connection to NATS."""
        mock_nc = AsyncMock()
        mock_nc.client_id = 12345
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        async def test():
            await self.client.connect()

            self.assertTrue(self.client.is_connected)
            self.assertEqual(self.client._nc, mock_nc)
            mock_connect.assert_called_once()

        asyncio.run(test())

    @patch('nats.connect')
    def test_connect_failure(self, mock_connect):
        """Test connection failure."""
        from nats.errors import NoServersError
        mock_connect.side_effect = NoServersError("No servers available")

        async def test():
            with self.assertRaises(NoServersError):
                await self.client.connect()

            self.assertFalse(self.client.is_connected)

        asyncio.run(test())

    @patch('nats.connect')
    def test_publish_message(self, mock_connect):
        """Test publishing a message."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        async def test():
            await self.client.connect()
            await self.client.publish("mesh.test", {"data": "test"})

            # Verify publish was called
            mock_nc.publish.assert_called_once()
            args = mock_nc.publish.call_args[0]
            subject = args[0]
            data = args[1]

            self.assertEqual(subject, "mesh.test")

            # Decode and verify message
            decoded = MeshMessage.decode(data)
            self.assertEqual(decoded.subject, "mesh.test")
            self.assertEqual(decoded.payload, {"data": "test"})
            self.assertEqual(decoded.sender, "test-node")

        asyncio.run(test())

    @patch('nats.connect')
    def test_broadcast_message(self, mock_connect):
        """Test broadcasting a message."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        async def test():
            await self.client.connect()
            await self.client.broadcast({"announcement": "hello mesh"})

            mock_nc.publish.assert_called_once()
            args = mock_nc.publish.call_args[0]
            subject = args[0]

            self.assertEqual(subject, SUBJECT_BROADCAST)

        asyncio.run(test())

    @patch('nats.connect')
    def test_send_to_node(self, mock_connect):
        """Test sending message to specific node."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        async def test():
            await self.client.connect()
            await self.client.send_to_node("target-node", {"message": "direct"})

            mock_nc.publish.assert_called_once()
            args = mock_nc.publish.call_args[0]
            subject = args[0]

            self.assertEqual(subject, "mesh.node.target-node")

        asyncio.run(test())

    @patch('nats.connect')
    def test_send_to_tier(self, mock_connect):
        """Test sending message to all nodes in a tier."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        async def test():
            await self.client.connect()
            await self.client.send_to_tier("T2", {"tier_message": "gpu nodes"})

            mock_nc.publish.assert_called_once()
            args = mock_nc.publish.call_args[0]
            subject = args[0]

            self.assertEqual(subject, "mesh.tier.T2")

        asyncio.run(test())

    @patch('nats.connect')
    def test_subscribe_and_receive(self, mock_connect):
        """Test subscribing to a subject and receiving messages."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        mock_sub = AsyncMock()
        mock_nc.subscribe.return_value = mock_sub

        received_messages = []

        async def test_handler(msg):
            received_messages.append(msg)

        async def test():
            await self.client.connect()
            await self.client.subscribe("mesh.test", test_handler)

            # Verify subscription was created
            mock_nc.subscribe.assert_called_once()
            args, kwargs = mock_nc.subscribe.call_args
            subject = args[0]
            callback = kwargs.get('cb', kwargs.get('callback'))

            self.assertEqual(subject, "mesh.test")

            # Simulate receiving a message
            test_msg = MeshMessage(
                subject="mesh.test",
                payload={"test": "data"},
                sender="sender-node"
            ).encode()

            mock_msg = MagicMock()
            mock_msg.data = test_msg
            mock_msg.reply = None

            await callback(mock_msg)

            # Verify handler was called
            self.assertEqual(len(received_messages), 1)
            self.assertEqual(received_messages[0].subject, "mesh.test")
            self.assertEqual(received_messages[0].payload, {"test": "data"})

        asyncio.run(test())

    @patch('nats.connect')
    def test_request_reply(self, mock_connect):
        """Test request/reply pattern."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        # Mock response
        response_msg = MeshMessage(
            subject="reply",
            payload={"result": "success"},
            sender="responder"
        )
        mock_response = MagicMock()
        mock_response.data = response_msg.encode()
        mock_nc.request.return_value = mock_response

        async def test():
            await self.client.connect()

            response = await self.client.request("mesh.service", {"query": "test"})

            self.assertIsNotNone(response)
            self.assertEqual(response.payload, {"result": "success"})
            self.assertEqual(response.sender, "responder")

        asyncio.run(test())

    @patch('nats.connect')
    def test_request_timeout(self, mock_connect):
        """Test request timeout."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_connect.return_value = mock_nc

        from nats.errors import TimeoutError
        mock_nc.request.side_effect = TimeoutError("Timeout")

        async def test():
            await self.client.connect()

            response = await self.client.request("mesh.service", {"query": "test"}, timeout=1.0)

            self.assertIsNone(response)

        asyncio.run(test())


class TestHeartbeatService(unittest.TestCase):
    """Test heartbeat service."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MagicMock()
        self.client.is_connected = True
        self.client.send_heartbeat = AsyncMock()

        self.node_info = {
            "name": "test-node",
            "tier": "T2",
            "address": "192.168.1.100"
        }

        self.heartbeat = HeartbeatService(
            client=self.client,
            node_info=self.node_info,
            interval=0.1  # Fast for testing
        )

    def test_start_stop(self):
        """Test starting and stopping heartbeat service."""
        async def test():
            await self.heartbeat.start()
            self.assertIsNotNone(self.heartbeat._task)

            await asyncio.sleep(0.2)  # Let it run a couple cycles

            await self.heartbeat.stop()
            self.assertIsNone(self.heartbeat._task)

            # Verify heartbeats were sent
            self.assertTrue(self.client.send_heartbeat.called)

        asyncio.run(test())


if __name__ == '__main__':
    unittest.main()