import certifi
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

from app.config import get_settings


_client: AsyncMongoClient | None = None


def get_client() -> AsyncMongoClient:
    global _client

    if _client is None:
        settings = get_settings()
        if not settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI is not configured")
        _client = AsyncMongoClient(
            settings.mongodb_uri,
            server_api=ServerApi("1"),
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10000,
        )

    return _client


def get_database():
    return get_client()[get_settings().mongodb_database]


def get_collection():
    settings = get_settings()
    return get_database()[settings.mongodb_collection]


def get_knowledge_collection():
    settings = get_settings()
    return get_database()[settings.mongodb_kb_collection]


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
