from typing import List, Dict, Any
from llm_playground_cli.client import LLMClient
from .interfaces import VectorDB, ReRanker
from llm_playground_cli.config import get_rag_config

class RAGEngine:
    def __init__(
        self, 
        client: LLMClient, 
        vector_db: VectorDB, 
        reranker: ReRanker, 
        collection_name: str = None,
        vector_size: int = None
    ):
        rag_config = get_rag_config()
        self.llm_client = client
        self.vector_db = vector_db
        self.reranker = reranker
        self.collection_name = collection_name or rag_config.collection_name
        vector_size = vector_size or rag_config.vector_size

         # Initialize DB collection
        self.vector_db.initialize(self.collection_name, vector_size)


    def ingest(self, documents: List[Dict[str, Any]]) -> int:
        """
        Ingest a list of documents. 
        Format: [{"content": "text...", "metadata": {"source": "file.md"}}]
        """
        if not documents:
            return 0
            
        texts = [doc["content"] for doc in documents]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.llm_client.get_embeddings(texts)
        
        payloads = [{"content": doc["content"], "metadata": doc.get("metadata", {})} for doc in documents]
            
        print("Upserting vectors to Vector DB...")
        self.vector_db.upsert(self.collection_name, embeddings, payloads)
        return len(texts)

    def search(self, query: str, top_k: int = 3, fetch_k: int = 15) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval:
        1. Fetch top `fetch_k` using fast Vector Search
        2. Re-rank them using a high-accuracy ReRanker
        """
        # 1. Semantic Search
        query_embedding = self.llm_client.get_embedding(query)
        search_results = self.vector_db.search(self.collection_name, query_embedding, fetch_k)
        
        if not search_results:
            return []
            
        # 2. Re-ranking
        ranked_results = self.reranker.rerank(query, search_results, top_k)
        
        return ranked_results