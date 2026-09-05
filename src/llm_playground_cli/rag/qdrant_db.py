import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .interfaces import VectorDB

class QdrantVectorDB(VectorDB):
    def __init__(self, path:str):
        self.client = QdrantClient(path=path)
    
    def initialize(self, collection_name: str, vector_size: int):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ))

    def upsert(self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        points = []
        for vec, payload in zip(vectors, payloads):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload
                )
            )
        self.client.upsert(collection_name=collection_name, points=points)


    def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        )
        # Unpack the payload so the orchestrator doesn't have to know about Qdrant's object structure
        return [
            {
                "content": hit.payload["content"], 
                "metadata": hit.payload.get("metadata", {})
            } 
            for hit in results.points
        ]

    def get_all_documents(self, collection_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve stored document payloads from collection using scroll."""
        if not self.client.collection_exists(collection_name):
            return []
        points, _ = self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return [
            {
                "content": p.payload.get("content", ""),
                "metadata": p.payload.get("metadata", {})
            }
            for p in points if p.payload and "content" in p.payload
        ]