"""NATS pub/sub messaging for inter-node communication."""
from .nats_client import MeshNATSClient, MeshMessage, HeartbeatService

__all__ = ["MeshNATSClient", "MeshMessage", "HeartbeatService"]
