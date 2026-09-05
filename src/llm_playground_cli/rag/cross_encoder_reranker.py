import os
import sys
from typing import List, Dict, Any
from .interfaces import ReRanker
from llm_playground_cli.config import get_rag_config
from sentence_transformers import CrossEncoder

class CrossEncoderReRanker(ReRanker):
    def __init__(self, model_name: str = None, max_length: int = None):
        rag_config = get_rag_config()
        model_name = model_name or rag_config.reranker_model_name
        max_length = max_length or rag_config.reranker_max_length
        
        # Suppress stderr/stdout prints (like unauthenticated HF Hub warnings and tqdm)
        stderr = sys.stderr
        try:
            with open(os.devnull, "w") as devnull:
                sys.stderr = devnull
                self.encoder = CrossEncoder(model_name, max_length=max_length)
        finally:
            sys.stderr = stderr

    def rerank(self, query: str, documents:List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        # The cross-encoder takes pairs of [query, document]
        candidates = [doc["content"] for doc in documents]
        cross_inp = [[query, cand] for cand in candidates]
        cross_scores = self.encoder.predict(cross_inp)
        
        ranked = []
        for i, doc in enumerate(documents):
            # Create a copy so we don't mutate the original dictionary
            doc_copy = dict(doc)
            doc_copy["score"] = float(cross_scores[i])
            ranked.append(doc_copy)
            
        # Sort by re-ranker score (highest first)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]