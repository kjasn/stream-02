import asyncio
import hashlib
import hmac
import inspect
import json
import os
import random
import struct
import time
import zlib
from hashlib import sha256

import aiohttp

# ── Proto 协议 ──────────────────────────────────────────────


class Proto:
    """B站开放平台长连协议包（大端对齐）"""

    HEADER_LEN = 16

    OP_HEARTBEAT = 2
    OP_HEARTBEAT_REPLY = 3
    OP_SEND_SMS_REPLY = 5
    OP_AUTH = 7
    OP_AUTH_REPLY = 8

    def __init__(self):
        self.packet_len = 0
        self.header_len = self.HEADER_LEN
        self.ver = 0
        self.op = 0
        self.seq = 0
        self.body = b""

    def pack(self) -> bytes:
        self.packet_len = self.HEADER_LEN + len(self.body)
        buf = struct.pack(">i", self.packet_len)
        buf += struct.pack(">h", self.HEADER_LEN)
        buf += struct.pack(">h", self.ver)
        buf += struct.pack(">i", self.op)
        buf += struct.pack(">i", self.seq)
        buf += self.body if isinstance(self.body, bytes) else self.body.encode()
        return buf

    def unpack(self, buf: bytes) -> int:
        """返回 consumed 字节数，便于递归解析"""
        if len(buf) < self.HEADER_LEN:
            return 0
        self.packet_len = struct.unpack(">i", buf[0:4])[0]
        self.header_len = struct.unpack(">h", buf[4:6])[0]
        self.ver = struct.unpack(">h", buf[6:8])[0]
        self.op = struct.unpack(">i", buf[8:12])[0]
        self.seq = struct.unpack(">i", buf[12:16])[0]
        if self.packet_len <= 0 or self.packet_len > len(buf):
            return 0
        self.body = buf[16 : self.packet_len]
        return self.packet_len

    def decode_body(self) -> list[dict]:
        """按 ver 解析 body，ver=2 时递归解压可能返回多个包"""
        if self.ver == 0:
            text = self.body.decode("utf-8")
            return [json.loads(text)] if text.strip() else []
        elif self.ver == 2:
            decompressed = zlib.decompress(self.body)
            results = []
            offset = 0
            while offset < len(decompressed):
                inner = Proto()
                n = inner.unpack(decompressed[offset:])
                if n == 0:
                    break
                offset += n
                results.extend(inner.decode_body())
            return results
        return []


# ── BiliLiveClient ──────────────────────────────────────────


class BiliLiveClient:
    CMD_DM = "LIVE_OPEN_PLATFORM_DM"
    CMD_DM_MIRROR = "LIVE_OPEN_PLATFORM_DM_MIRROR"
    CMD_GIFT = "LIVE_OPEN_PLATFORM_SEND_GIFT"
    CMD_SUPER_CHAT = "LIVE_OPEN_PLATFORM_SUPER_CHAT"
    CMD_SUPER_CHAT_DEL = "LIVE_OPEN_PLATFORM_SUPER_CHAT_DEL"
    CMD_GUARD = "LIVE_OPEN_PLATFORM_GUARD"
    CMD_LIKE = "LIVE_OPEN_PLATFORM_LIKE"
    CMD_ROOM_ENTER = "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER"
    CMD_LIVE_START = "LIVE_OPEN_PLATFORM_LIVE_START"
    CMD_LIVE_END = "LIVE_OPEN_PLATFORM_LIVE_END"
    CMD_INTERACTION_END = "LIVE_OPEN_PLATFORM_INTERACTION_END"

    def __init__(
        self,
        id_code: str,
        app_id: int,
        key: str,
        secret: str,
        host: str = "https://live-open.biliapi.com",
    ):
        self.id_code = id_code
        self.app_id = app_id
        self.key = key
        self.secret = secret
        self.host = host
        self.game_id = ""
        self._handlers: dict[str, list] = {}
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._stop_event = asyncio.Event()

    # ── 事件回调 ──

    def on(self, cmd: str):
        """装饰器：注册事件处理函数"""

        def decorator(fn):
            self._handlers.setdefault(cmd, []).append(fn)
            return fn

        return decorator

    async def _emit(self, cmd: str, data: dict):
        for fn in self._handlers.get(cmd, []):
            try:
                if inspect.iscoroutinefunction(fn):
                    await fn(data)
                else:
                    fn(data)
            except Exception as e:
                print(f"[handler error] cmd={cmd}: {e}")

    # ── 停止 ──

    async def stop(self) -> None:
        """Graceful stop: signal loops to exit and close the WebSocket."""
        self._stop_event.set()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    # ── HTTP 签名 ──

    def _sign(self, params: str) -> dict:
        md5 = hashlib.md5(params.encode()).hexdigest()
        ts = str(int(time.time()))
        nonce = str(random.randint(1, 100000) + time.time())
        header_map = {
            "x-bili-timestamp": ts,
            "x-bili-signature-method": "HMAC-SHA256",
            "x-bili-signature-nonce": nonce,
            "x-bili-accesskeyid": self.key,
            "x-bili-signature-version": "1.0",
            "x-bili-content-md5": md5,
        }
        sorted_keys = sorted(header_map)
        header_str = "\n".join(f"{k}:{header_map[k]}" for k in sorted_keys)
        sig = hmac.new(self.secret.encode(), header_str.encode(), sha256).hexdigest()
        header_map["Authorization"] = sig
        header_map["Content-Type"] = "application/json"
        header_map["Accept"] = "application/json"
        return header_map

    # ── 获取长连信息 ──

    async def _get_websocket_info(self) -> tuple[str, str]:
        params = json.dumps({"code": self.id_code, "app_id": self.app_id})
        headers = self._sign(params)

        assert self._session is not None
        async with self._session.post(f"{self.host}/v2/app/start", headers=headers, data=params) as response:
            data = await response.json()
            print("[API] /v2/app/start:", json.dumps(data, ensure_ascii=False))
            self.game_id = data["data"]["game_info"]["game_id"]
            ws_info = data["data"]["websocket_info"]
            return ws_info["wss_link"][0], ws_info["auth_body"]

    # ── 应用心跳 ──

    async def _app_heartbeat(self):
        while not self._stop_event.is_set():
            assert self._session is not None
            await asyncio.sleep(20)
            url = f"{self.host}/v2/app/heartbeat"
            params = json.dumps({"game_id": self.game_id})
            headers = self._sign(params)
            try:
                async with self._session.post(url, headers=headers, data=params) as response:
                    await response.json()
                    print("[appHeartbeat] success")
            except Exception as e:
                print(f"[appHeartbeat] fail: {e}")

    # ── 结束应用 ──

    async def _end_app(self):
        url = f"{self.host}/v2/app/end"
        params = json.dumps({"game_id": self.game_id, "app_id": self.app_id})
        headers = self._sign(params)
        try:
            async with self._session.post(  # type: ignore
                url, headers=headers, data=params
            ) as response:
                result = await response.json()
                print("[API] /v2/app/end:", result)
        except Exception as e:
            print(f"[endApp] fail: {e}")

    # ── 连接 ──

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        addr, auth_body = await self._get_websocket_info()
        print(f"[connect] {addr}")

        assert self._session is not None
        # 使用 aiohttp 的 WebSocket 连接
        ws = await self._session.ws_connect(
            addr,
            ssl=False,  # 禁用 SSL 验证
            autoping=True,  # 自动响应 ping
        )

        # 鉴权
        req = Proto()
        req.op = Proto.OP_AUTH
        req.body = auth_body.encode()
        await ws.send_bytes(req.pack())  # 注意：使用 send_bytes 发送二进制数据

        # 接收鉴权回复
        msg = await ws.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            resp = Proto()
            resp.unpack(msg.data)
            reply = json.loads(resp.body)
            if reply.get("code") == 0:
                print("[auth] 成功")
            else:
                print(f"[auth] 失败: {reply}")
        else:
            print(f"[auth] 异常: {msg.type}")

        return ws

    # ── 主循环 ──

    async def _recv_loop(self):
        print("[recvLoop] start")
        while not self._stop_event.is_set():
            assert self._ws is not None
            msg = await self._ws.receive()

            if msg.type == aiohttp.WSMsgType.BINARY:
                pkt = Proto()
                pkt.unpack(msg.data)

                if pkt.op == Proto.OP_HEARTBEAT_REPLY:
                    continue  # 心跳回复，忽略

                elif pkt.op == Proto.OP_SEND_SMS_REPLY:
                    for msg_data in pkt.decode_body():
                        cmd = msg_data.get("cmd", "")
                        await self._emit(cmd, msg_data.get("data", {}))
                        await self._emit("*", msg_data)

            elif msg.type == aiohttp.WSMsgType.PING:
                # aiohttp 会自动回复 pong，但为了保险也可以手动处理
                await self._ws.pong()

            elif msg.type == aiohttp.WSMsgType.CLOSE:
                print("[recvLoop] 连接关闭")
                break

            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"[recvLoop] 错误: {self._ws.exception()}")
                break

    async def _heartbeat(self):
        while not self._stop_event.is_set():
            assert self._ws is not None

            await asyncio.sleep(20)
            pkt = Proto()
            pkt.op = Proto.OP_HEARTBEAT
            try:
                await self._ws.send_bytes(pkt.pack())
                print("[heartBeat] success")
            except Exception as e:
                print(f"[heartBeat] fail: {e}")
                break

    # ── 入口 ──

    async def run(self):
        """Connect and run. Supports graceful stop() and task-level error isolation."""
        self._stop_event.clear()
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self._session = session
            self._ws = await self._connect()

            async def _safe(name, coro):
                try:
                    await coro
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[run] {name} error: {e}")

            try:
                await asyncio.gather(
                    _safe("recv_loop", self._recv_loop()),
                    _safe("heartbeat", self._heartbeat()),
                    _safe("app_heartbeat", self._app_heartbeat()),
                )
            except Exception as e:
                print(f"[run] 异常: {e}")
            finally:
                await self._end_app()
                if self._ws and not self._ws.closed:
                    await self._ws.close()

    async def run_with_reconnect(self, max_attempts: int = 0, delay: float = 5.0):
        """Run with automatic reconnection on failure.

        max_attempts=0 means unlimited retries.
        """
        attempt = 0
        while not self._stop_event.is_set():
            try:
                await self.run()
            except Exception as e:
                print(f"[run] connection failed (attempt {attempt + 1}): {e}")
            attempt += 1
            if max_attempts > 0 and attempt >= max_attempts:
                print("[run] max reconnection attempts reached")
                break
            if not self._stop_event.is_set():
                await asyncio.sleep(delay)


async def bili_client():
    from dotenv import load_dotenv

    # 加载 .env 文件
    load_dotenv()

    client = BiliLiveClient(
        id_code=os.getenv("BILI_ID_CODE", ""),
        app_id=int(os.getenv("BILI_APP_ID", "0")),
        key=os.getenv("BILI_KEY", ""),
        secret=os.getenv("BILI_SECRET", ""),
    )

    # 弹幕回调
    @client.on(BiliLiveClient.CMD_DM)
    def on_danmaku(data):
        print(f"弹幕 [{data.get('uname')}]: {data.get('msg')}")

    @client.on(BiliLiveClient.CMD_SUPER_CHAT)
    def on_superchat(data):
        print(f"价值 {data.get('rmb')} 的 superchat: {data.get('message')}")

    @client.on(BiliLiveClient.CMD_GIFT)
    def on_gift(data):
        print(f"礼物 [{data.get('uname')}]: {data.get('gift_name')} x{data.get('gift_num')}")

    @client.on(BiliLiveClient.CMD_GUARD)
    def on_guard(data):
        user = data.get("user_info", {})
        level = {1: "总督", 2: "提督", 3: "舰长"}.get(data.get("guard_level"), "?")
        print(f"大航海 [{user.get('uname')}]: 开通了{level}")

    @client.on(BiliLiveClient.CMD_LIKE)
    def on_like(data):
        print(f"{data.get('like_text')} x{data.get('like_count')}")

    @client.on(BiliLiveClient.CMD_ROOM_ENTER)
    def on_enter(data):
        print(f"{data.get('uname')} 进入房间")

    @client.on("*")
    def on_all(data):
        print(f"[msg] {data.get('cmd', 'unknown')}: {json.dumps(data, ensure_ascii=False)[:200]}")

    asyncio.run(client.run())
