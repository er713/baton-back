from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        print("Initialized ConnectionManager")
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # print("Connected with somebody")
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        print("\tINSIDE sending messagepersonal ", type(message), message)
        await websocket.send_json({"results": message})

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
