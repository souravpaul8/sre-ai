from pathlib import Path
from rich.console import Console
from rich.text import Text
import time

from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import GenerationConfig


console = Console()

# Gemini 2.0 Flash pricing (per million tokens)
# Update these if you switch models
COST_PER_MILLION_INPUT = 0.10   # $0.10 / 1M input tokens
COST_PER_MILLION_OUTPUT = 0.40  # $0.40 / 1M output tokens

def estimate_cost(input_tokens: int, output_tokens: 
    int) -> float:
    input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT
    return round(input_cost + output_cost, 6)

def run(file_path: str, config:GenerationConfig,
    override_temp: float | None = None) -> None:
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return
    
    content = path.read_text(encoding="utf-8")

    client = LLMClient(config)
    input_tokens = client.count_tokens(content)
    console.print(
        f"\n[bold cyan]Analyzing[/bold cyan] [yellow]{path.name}[/yellow] "
        f"([green]{input_tokens:,}[/green] tokens)...\n"
    )

    system_prompt = (
        "You are an expert SRE. Analyze the provided file and give a clear, "
        "concise summary. Focus on: what it does, key configurations, and any "
        "notable patterns or concerns. Be specific and technical."
    )

    user_prompt = f"Please summarize the following file ({path.name}):\n\n{content}"

    output_chunks = []
    start = time.time()

    for chunk in client.stream(user_prompt,system=system_prompt,override_temp=override_temp):
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

