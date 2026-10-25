import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..websocket_manager import ws_manager

logger = logging.getLogger("spendwise.websocket")
router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send initial connection acknowledgment
        await websocket.send_text(json.dumps({
            "type": "CONNECTION_ESTABLISHED",
            "message": "Connected to SpendWise Live real-time broadcast engine.",
            "status": "connected"
        }))

        while True:
            # Keep listening for client heartbeats or messages
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                if data.get("type") == "PING":
                    await websocket.send_text(json.dumps({"type": "PONG"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection error: {e}")
        await ws_manager.disconnect(websocket)
