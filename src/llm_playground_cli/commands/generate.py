from pathlib import Path
from rich.console import Console
import time

from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import GenerationConfig

console = Console()

COST_PER_MILLION_INPUT = 0.10
COST_PER_MILLION_OUTPUT = 0.40


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT
    return round(input_cost + output_cost, 6)


def _gather_project_context(project_dir: Path) -> str:
    SKIP_DIRS = {".venv", ".git", "__pycache__", "dist", "build", ".eggs"}
    SKIP_EXTENSIONS = {".pyc", ".pyo", ".so", ".lock", ".env"}
    MAX_FILE_SIZE = 4000  # chars — avoid sending huge files

    context_parts = ["# Project Context\n"]
    context_parts.append(f"**Project root:** `{project_dir.name}`\n\n")
    context_parts.append("## File Tree\n```\n")

    # Build a simple file tree
    for path in sorted(project_dir.rglob("*")):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.is_file():
            rel = path.relative_to(project_dir)
            context_parts.append(f"{rel}\n")

    context_parts.append("```\n\n## File Contents\n\n")

    # Read content of key files
    PRIORITY_FILES = {"pyproject.toml", "requirements.txt", "Makefile", "Dockerfile"}
    PY_EXTENSIONS = {".py", ".yaml", ".yml", ".toml", ".md", ".sh", ".txt"}

    for path in sorted(project_dir.rglob("*")):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if not path.is_file():
            continue
        if path.suffix in SKIP_EXTENSIONS:
            continue
        if path.suffix not in PY_EXTENSIONS and path.name not in PRIORITY_FILES:
            continue

        try:
            content = path.read_text(encoding="utf-8")
            if len(content) > MAX_FILE_SIZE:
                content = content[:MAX_FILE_SIZE] + "\n... (truncated)"
            rel = path.relative_to(project_dir)
            context_parts.append(f"### `{rel}`\n```\n{content}\n```\n\n")
        except Exception:
            pass  # Skip unreadable files silently

    return "".join(context_parts)


def run(
    project_dir: str = ".",
    config: GenerationConfig = None,
    override_temp: float | None = None,
) -> None:

    path = Path(project_dir).resolve()
    client = LLMClient(config)

    console.print(
        f"\n[bold cyan]Generating README[/bold cyan] for "
        f"[yellow]{path.name}[/yellow]...\n"
    )

    context = _gather_project_context(path)
    input_tokens = client.count_tokens(context)
    console.print(f"[dim]Context: {input_tokens:,} tokens[/dim]\n")

    system_prompt = (
        "You are a principal software engineer who writes excellent README files. "
        "Generate a professional, comprehensive README.md based on the project context. "
        "Include: project title, description, features, installation, usage examples, "
        "and project structure. Use clear markdown formatting."
    )

    user_prompt = (
        f"Generate a README.md for this project:\n\n{context}"
    )

    output_chunks = []
    start = time.time()

    for chunk in client.stream(user_prompt, system=system_prompt, override_temp=override_temp):
        print(chunk, end="", flush=True)
        output_chunks.append(chunk)

    latency = round(time.time() - start, 2)
    print()

    full_response = "".join(output_chunks)
    output_tokens = client.count_tokens(full_response)
    cost = estimate_cost(input_tokens, output_tokens)

    console.print(
        f"\n[dim][Tokens: {input_tokens:,} in / {output_tokens:,} out | "
        f"Cost: ${cost:.4f} | Latency: {latency}s][/dim]"
    )
