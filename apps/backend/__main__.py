"""Entry point: python -m backend"""

import asyncio

from services.bili_client import bili_client

if __name__ == "__main__":
    asyncio.run(bili_client())
