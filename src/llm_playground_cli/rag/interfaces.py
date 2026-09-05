from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDB(ABC):
    @abstractmethod
    def initialize(self, collection_name: str, vector_size: int):
        """Ensure the collection/table exists."""
        pass
    
    @abstractmethod
    def upsert(self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Insert or update vectors and their payloads."""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Search and return the payloads of the closest matches."""
        pass

    @abstractmethod
    def get_all_documents(self, collection_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve stored document payloads from collection."""
        pass

class ReRanker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents:List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Take a query and a list of documents, and return the top_k sorted by relevance."""
        pass