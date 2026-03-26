"""Tests for MYCONEX service discovery (mDNS hub service auto-discovery)."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.discovery.mesh_discovery import ServiceURLs, ServiceDiscoveryError


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


if __name__ == '__main__':
    unittest.main()
