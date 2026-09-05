import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
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

        # In-memory BM25 index cache
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenization."""
        return re.findall(r"\w+", text.lower())

    def _build_bm25_index(self):
        """Build or refresh BM25 index from documents stored in VectorDB."""
        docs = self.vector_db.get_all_documents(self.collection_name)
        if docs:
            tokenized_corpus = [self._tokenize(doc["content"]) for doc in docs]
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_docs = docs
        else:
            self._bm25 = None
            self._bm25_docs = []

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

        # Refresh BM25 index with updated documents
        self._build_bm25_index()
        return len(texts)

    def search_keyword(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        if self._bm25 is None or not self._bm25_docs:
            self._build_bm25_index()

        if not self._bm25 or not self._bm25_docs:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self._bm25.get_scores(tokenized_query)
        # Pair with documents and filter zero scores
        scored_pairs = [
            (score, doc) for score, doc in zip(doc_scores, self._bm25_docs) if score > 0
        ]
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_pairs[:limit]]

    def search(self, query: str, top_k: int = 3, fetch_k: int = 15) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval:
        1. Fetch candidates from Semantic Vector Search
        2. Fetch candidates from Keyword (BM25) Search
        3. Merge unique candidates and re-rank with CrossEncoder
        """
        # 1. Semantic Vector Search
        query_embedding = self.llm_client.get_embedding(query)
        vector_results = self.vector_db.search(self.collection_name, query_embedding, fetch_k)

        # 2. Keyword BM25 Search
        keyword_results = self.search_keyword(query, limit=fetch_k)

        # 3. Deduplicate candidates by content
        seen_contents = set()
        candidate_docs = []
        for doc in vector_results + keyword_results:
            content = doc.get("content", "").strip()
            if content and content not in seen_contents:
                seen_contents.add(content)
                candidate_docs.append(doc)

        if not candidate_docs:
            return []

        # 4. Cross-Encoder Re-ranking
        ranked_results = self.reranker.rerank(query, candidate_docs, top_k)
        return ranked_results