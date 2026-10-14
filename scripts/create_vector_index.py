from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from app.config import get_settings


settings = get_settings()

client = MongoClient(settings.mongodb_uri)
collection = client[settings.mongodb_database][settings.mongodb_kb_collection]

model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": settings.gemini_embedding_dimensions,
                "similarity": "cosine",
            }
        ]
    },
    name=settings.mongodb_vector_index,
    type="vectorSearch",
)

try:
    result = collection.create_search_index(model=model)
    print(f"Created vector search index: {result}")
finally:
    client.close()
