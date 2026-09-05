import os
from rich.console import Console
from llm_playground_cli.config import GenerationConfig, get_default_config
from llm_playground_cli.client import LLMClient
from llm_playground_cli.rag.orchestrator import RAGEngine
from llm_playground_cli.rag.qdrant_db import QdrantVectorDB
from llm_playground_cli.rag.cross_encoder_reranker import CrossEncoderReRanker
from llm_playground_cli.rag.loader import prepare_documents

console = Console()

def run(
    config: GenerationConfig, 
    file_path: str = None, 
    text: str = None,
    collection_name: str = "sre_docs_v2"
) -> None:
    """Ingest a file (.pdf, .md, .txt) or raw text into Qdrant."""
    if not file_path and not text:
        console.print("[red]Error: You must provide either a file path (--file) or raw text (--text)[/red]")
        return

    try:
        docs = prepare_documents(text=text, file_path=file_path)
        if not docs:
            console.print("[yellow]Warning: No text content found to ingest.[/yellow]")
            return

        source_desc = f"file '{file_path}'" if file_path else "provided text"
        console.print(f"[cyan]Loaded {len(docs)} chunk(s) from {source_desc}...[/cyan]")

        # Setup RAG Engine
        client = LLMClient(config)
        qdrant_path = os.path.join(os.path.expanduser("~"), ".sre-ai-rag.db")
        vector_db = QdrantVectorDB(path=qdrant_path)
        reranker = CrossEncoderReRanker()

        engine = RAGEngine(
            client=client,
            vector_db=vector_db,
            reranker=reranker,
            collection_name=collection_name
        )

        with console.status("[bold green]Generating embeddings & upserting into Qdrant...[/bold green]"):
            count = engine.ingest(docs)

        console.print(f"[bold green]✓ Successfully ingested {count} chunk(s) into collection '{collection_name}'![/bold green]")

    except Exception as e:
        console.print(f"[red]Failed to ingest: {e}[/red]")
