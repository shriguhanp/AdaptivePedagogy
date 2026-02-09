"""
Progress Broadcaster - Manages WebSocket broadcasting of knowledge base progress
"""

import asyncio
from typing import Optional

from fastapi import WebSocket


class ProgressBroadcaster:
    """Manages WebSocket broadcasting of knowledge base progress"""

    _instance: Optional["ProgressBroadcaster"] = None
    _connections: dict[str, set[WebSocket]] = {}  # kb_name -> Set[WebSocket]
    _lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "ProgressBroadcaster":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self, kb_name: str, websocket: WebSocket):
        """Connect WebSocket to specified knowledge base"""
        async with self._lock:
            if kb_name not in self._connections:
                self._connections[kb_name] = set()
            self._connections[kb_name].add(websocket)
            print(
                f"[ProgressBroadcaster] Connected WebSocket for KB '{kb_name}' (total: {len(self._connections[kb_name])})"
            )

    async def disconnect(self, kb_name: str, websocket: WebSocket):
        """Disconnect WebSocket connection"""
        async with self._lock:
            if kb_name in self._connections:
                self._connections[kb_name].discard(websocket)
                if not self._connections[kb_name]:
                    del self._connections[kb_name]
                print(f"[ProgressBroadcaster] Disconnected WebSocket for KB '{kb_name}'")

    async def broadcast(self, kb_name: str, progress: dict):
        """Broadcast progress update to all WebSocket connections for specified knowledge base"""
        async with self._lock:
            if kb_name not in self._connections:
                return

            # Create list of connections to remove (closed connections)
            to_remove = []

            for websocket in self._connections[kb_name]:
                try:
                    await websocket.send_json({"type": "progress", "data": progress})
                except Exception as e:
                    # Connection closed or error, mark for removal
                    print(
                        f"[ProgressBroadcaster] Error sending to WebSocket for KB '{kb_name}': {e}"
                    )
                    to_remove.append(websocket)

            # Remove closed connections
            for ws in to_remove:
                self._connections[kb_name].discard(ws)

            if not self._connections[kb_name]:
                del self._connections[kb_name]

    def get_connection_count(self, kb_name: str) -> int:
        """Get connection count for specified knowledge base"""
        return len(self._connections.get(kb_name, set()))
