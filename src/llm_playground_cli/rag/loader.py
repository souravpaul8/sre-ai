import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

def chunk_text(
    text: str, 
    is_markdown: bool = False, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> List[str]:
    """Intelligently split text using langchain splitters."""
    if not text.strip():
        return []

    if is_markdown:
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    return splitter.split_text(text)


def load_file_content(file_path: str) -> tuple[str, bool]:
    """
    Read content from .txt, .md, or .pdf.
    Returns a tuple of (content, is_markdown).
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    
    if ext == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required to parse PDFs. Please run 'pip install pypdf'")
            
        reader = pypdf.PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n\n".join(pages_text), False
    
    elif ext == ".md":
        return path.read_text(encoding="utf-8", errors="replace"), True
    else:
        # Default text fallback (.txt, .yaml, .json, etc.)
        return path.read_text(encoding="utf-8", errors="replace"), False


def prepare_documents(
    text: Optional[str] = None, 
    file_path: Optional[str] = None, 
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Prepare list of document dicts suitable for RAGEngine.ingest():
    [{"content": "...", "metadata": {"source": "..."}}]
    """
    is_markdown = False
    if file_path:
        raw_text, is_markdown = load_file_content(file_path)
        source = Path(file_path).name
    elif text:
        raw_text = text
        source = "cli_text_input"
    else:
        raise ValueError("Either 'text' or 'file_path' must be provided.")

    chunks = chunk_text(
        raw_text, 
        is_markdown=is_markdown, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    docs = [
        {
            "content": chunk,
            "metadata": {
                "source": source,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
        }
        for idx, chunk in enumerate(chunks)
    ]
    return docs