import redis
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class StateManager:
    """
    Redis-backed state manager to handle low-latency communication 
    between Data Feed, Risk, and Order Execution modules.
    """
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def set(self, key: str, value: Any) -> bool:
        """Stores a JSON serializable value in Redis."""
        try:
            return self.client.set(key, json.dumps(value))
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes a value from Redis."""
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def publish(self, channel: str, message: Any) -> int:
        """Publishes a message to a Redis channel."""
        try:
            return self.client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error(f"Error publishing to {channel}: {e}")
            return 0

    def get_subscriber(self):
        """Returns a Redis PubSub object for listening to channels."""
        return self.client.pubsub()
