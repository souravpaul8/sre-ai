import subprocess
import os
from pathlib import Path
from llm_playground_cli.tools.base import Tool, ToolResult
from llm_playground_cli.config import get_default_config
from llm_playground_cli.client import LLMClient
from llm_playground_cli.rag.orchestrator import RAGEngine
from llm_playground_cli.rag.qdrant_db import QdrantVectorDB
from llm_playground_cli.rag.cross_encoder_reranker import CrossEncoderReRanker

# Global instance so we don't reload the models/DB on every tool call
_rag_engine = None

def _search_knowledge_base(query: str) -> ToolResult:
    """Search the vector database."""
    global _rag_engine
    try:
        if _rag_engine is None:
            client = LLMClient(get_default_config())
            
            # Setup dependencies for the Orchestrator
            qdrant_path = os.path.join(os.path.expanduser("~"), ".sre-ai-rag.db")
            vector_db = QdrantVectorDB(path=qdrant_path)
            reranker = CrossEncoderReRanker()
            
            _rag_engine = RAGEngine(
                client=client, 
                vector_db=vector_db, 
                reranker=reranker
            )
            
        results = _rag_engine.search(query, top_k=3)
        if not results:
            return ToolResult(success=True, data="No relevant information found in the knowledge base.")
        
        formatted = []
        for i, res in enumerate(results):
            source = res['metadata'].get('source', 'Unknown')
            formatted.append(f"--- Result {i+1} (Source: {source}) ---\n{res['content']}")
            
        return ToolResult(success=True, data="\n\n".join(formatted))
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))


def _read_file(file_path: str) -> ToolResult:
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return ToolResult(success=False, data="", error=f"File not found: {file_path}")
        if not path.is_file():
            return ToolResult(success=False, data="", error=f"Path is not a file: {file_path}")
        
        content = path.read_text(encoding="utf-8", errors="replace")

        max_chars = 8000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[... truncated, showing first {max_chars} chars ...]"
        return ToolResult(success=True, data=content)
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))

def _list_files(directory: str) -> ToolResult:
    """List contents of a directory."""
    try:
        path = Path(directory).expanduser().resolve()
        if not path.exists():
            return ToolResult(success=False, data="", error=f"Directory not found: {directory}")
        if not path.is_dir():
            return ToolResult(success=False, data="", error=f"Not a directory: {directory}")
        entries = []
        for item in sorted(path.iterdir()):
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else ""
            entries.append(f"  {kind:<5} {size:>8}  {item.name}")
        result = f"Contents of {path}:\n" + "\n".join(entries)
        return ToolResult(success=True, data=result)
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))

def _run_shell(command: str) -> ToolResult:
    """
    Run a shell command and return its output.
    DANGEROUS — requires user confirmation in the agent loop.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,       # needed for shell commands — agent confirms first
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd(),
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            return ToolResult(
                success=False,
                data=output,
                error=f"Exit code {result.returncode}: {stderr}",
            )
        # Truncate long output
        if len(output) > 5000:
            output = output[:5000] + "\n\n[... truncated ...]"
        return ToolResult(success=True, data=output or "(no output)")
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, data="", error="Command timed out after 30 seconds")
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))

# ── Tool Definitions ─────────────────────────────────────────
# The descriptions below are PROMPTS — the LLM reads them to decide
# when to call each tool. Write them like you're explaining to a coworker.
read_file_tool = Tool(
    name="read_file",
    description=(
        "Read the contents of a file from disk. "
        "Use this when you need to see the contents of a config file, "
        "log file, script, YAML, Dockerfile, or any text file. "
        "Returns the file contents as a string. "
        "Do NOT use this for listing directory contents — use list_files instead."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "file_path": {
                "type": "STRING",
                "description": "Absolute or relative path to the file to read.",
            },
        },
        "required": ["file_path"],
    },
    function=_read_file,
)
list_files_tool = Tool(
    name="list_files",
    description=(
        "List the files and directories inside a given directory. "
        "Use this to explore the project structure or find files. "
        "Returns a list of entries with their type (file/dir) and size. "
        "Do NOT use this to read file contents — use read_file instead."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "directory": {
                "type": "STRING",
                "description": "Path to the directory to list. Defaults to current directory.",
            },
        },
        "required": ["directory"],
    },
    function=_list_files,
)
run_shell_tool = Tool(
    name="run_shell",
    description=(
        "Execute a shell command and return its stdout/stderr output. "
        "Use this for any command-line operation: git, grep, find, curl, etc. "
        "WARNING: This runs real commands on the user's machine. "
        "Always prefer specific tools (read_file, list_files, get_pod_status) "
        "over run_shell when a specific tool exists for the task. "
        "Only use run_shell for commands that don't have a dedicated tool."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "command": {
                "type": "STRING",
                "description": "The shell command to execute (e.g. 'ls -la', 'git status').",
            },
        },
        "required": ["command"],
    },
    function=_run_shell,
    is_dangerous=True,   # ← requires user confirmation
)
search_kb_tool = Tool(
    name="search_knowledge_base",
    description=(
        "Search the local knowledge base and documentation for answers. "
        "Use this tool when the user asks a conceptual question, how to do something, "
        "or asks about internal guidelines, runbooks, or specific architectures."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search query to look up in the vector database.",
            },
        },
        "required": ["query"],
    },
    function=_search_knowledge_base,
)

ALL_BUILTIN_TOOLS = [read_file_tool, list_files_tool, run_shell_tool, search_kb_tool]