import asyncio
import json
import logging
from typing import Set, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger("spendwise.websocket")


class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Broadcasts a JSON event payload to all currently connected clients."""
        payload = {
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        text_data = json.dumps(payload)

        async with self._lock:
            stale_connections = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(text_data)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}. Marking for cleanup.")
                    stale_connections.append(connection)

            for stale in stale_connections:
                self.active_connections.discard(stale)

    async def broadcast_transaction_added(self, transaction: Dict[str, Any], month_summary: Dict[str, Any], diff_info: Dict[str, Any]):
        await self.broadcast("TRANSACTION_ADDED", {
            "transaction": transaction,
            "month_summary": month_summary,
            "diff_info": diff_info,
        })

    async def broadcast_models_retrained(self, retrain_payload: Dict[str, Any]):
        await self.broadcast("MODELS_RETRAINED", retrain_payload)

    async def broadcast_overspend_alert(self, alert_payload: Dict[str, Any]):
        await self.broadcast("OVERSPEND_ALERT", alert_payload)


ws_manager = WebSocketManager()
