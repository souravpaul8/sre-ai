import os
from llm_playground_cli.config import get_default_config
from llm_playground_cli.client import LLMClient
from llm_playground_cli.rag.orchestrator import RAGEngine
from llm_playground_cli.rag.qdrant_db import QdrantVectorDB
from llm_playground_cli.rag.cross_encoder_reranker import CrossEncoderReRanker

# 1. Initialize components
client = LLMClient(get_default_config())
qdrant_path = os.path.join(os.path.expanduser("~"), ".sre-ai-rag.db")
vector_db = QdrantVectorDB(path=qdrant_path)
reranker = CrossEncoderReRanker()

# 2. Wire them into the Orchestrator
engine = RAGEngine(
    client=client,
    vector_db=vector_db,
    reranker=reranker
)

# 3. Dummy Runbooks for testing
docs = [
    {
        "content": "To restart the payment service in production, you must first drain the connections using the /drain endpoint, then run kubectl rollout restart deployment/payment-service -n production.",
        "metadata": {"source": "runbook-payments.md"}
    },
    {
        "content": "If you see exit code 137, the pod was OOMKilled. You need to increase the memory limits in the deployment spec and re-apply.",
        "metadata": {"source": "troubleshooting.md"}
    }
]

print("Ingesting documents into Qdrant...")
engine.ingest(docs)

print("\nSearching for: 'How do I safely restart payments?'")
results = engine.search("How do I safely restart payments?", top_k=1)
for res in results:
    print(f"\nScore: {res['score']:.4f}")
    print(f"Content: {res['content']}")
