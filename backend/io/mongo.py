import logging
from typing import Optional

from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure
from typing_extensions import Self

from backend.common.config import get_settings

logger = logging.getLogger("io")


class MongoDB:
    _instance: Optional["MongoDB"] = None
    _client: Optional[AsyncMongoClient] = None
    _db_name: str = "app"

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # pyright: ignore[reportReturnType]

    async def connect(self):
        _config = get_settings()
        url = _config.db.mongo_url
        self._db_name = _config.db.mongo_db_name

        self._client = AsyncMongoClient(
            url,
            maxPoolSize=100,  # 最大连接池
            minPoolSize=10,  # 最下连接池
            maxIdleTimeMS=60000,  # 最大空闲时间
            connectTimeoutMS=5000,  # 连接超时
            socketTimeoutMs=30000,  # Socket超时
            serverSelectionTimeoutMS=5000,  # 服务器选择超时
            retryWrites=True,  # 自动重试写操作
            retryReads=True,  # 自动重试读操作
            heartbeatFrequencyMS=10000,
            appname="backend",
        )

        # 测试异步连接
        try:
            await self._client.admin.command("ping")
            logger.info(f"PyMongo连接成功, 数据库: {self._db_name}")
        except ConnectionFailure as e:
            logger.error(f"PyMongo连接失败: {e}")
            raise

        return self._client[self._db_name]

    async def close(self):
        if self._client is not None:
            await self._client.close()
            logger.info("PyMongo连接关闭")

    @property
    def client(self) -> AsyncMongoClient:
        if self._client is None:
            raise RuntimeError("MongoDB 未连接")
        return self._client

    @property
    def db(self):
        if self._client is None:
            raise RuntimeError("MongoDB 未连接")
        return self._client[self._db_name]


mongo = MongoDB()
