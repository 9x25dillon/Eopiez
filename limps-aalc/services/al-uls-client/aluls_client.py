import json, asyncio, websockets, httpx

class ALULSClient:
    def __init__(self, base_http: str = "http://julia-ref:8008", ws_url: str = "ws://julia-ref:8008/ws"):
        self.base_http = base_http
        self.ws_url = ws_url

    async def simplify_ws(self, expr: str):
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps({"fn":"simplify","expr": expr}))
            msg = await ws.recv()
            return json.loads(msg)

    async def batch_simplify_http(self, exprs):
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{self.base_http}/simplify", json={"exprs": exprs})
            r.raise_for_status()
            return r.json()