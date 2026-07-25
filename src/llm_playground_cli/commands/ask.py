# src/llm_playground_cli/commands/ask.py

import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import GenerationConfig
from llm_playground_cli.prompts.loader import library
from llm_playground_cli.prompts.router import QueryRouter

console = Console()

COST_PER_MILLION_INPUT = 0.10
COST_PER_MILLION_OUTPUT = 0.40


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT +
        (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT,
        6
    )


def _show_fallback_menu() -> str:
    templates = library.list_templates()
    console.print("\n[yellow]Could not auto-detect domain. Please choose:[/yellow]\n")
    for i, t in enumerate(templates, 1):
        console.print(f"  [cyan]{i}[/cyan]. {t['name']:<15} — {t['description']}")

    choice = Prompt.ask(
        "\nExpert",
        choices=[str(i) for i in range(1, len(templates) + 1)],
    )
    return templates[int(choice) - 1]["name"]


def run(config: GenerationConfig, cmd_question: str | None = None) -> None:
    console.print("\n[bold cyan]SRE AI Assistant[/bold cyan] — ask any technical question\n")
    
    if cmd_question:
        question = cmd_question
        console.print(f"[bold]Your question:[/bold] {question}")
    else:
        question = Prompt.ask("[bold]Your question[/bold]")

    if not question.strip():
        console.print("[red]No question entered.[/red]")
        return


    console.print("\n[dim]Routing your question...[/dim]")
    router = QueryRouter()
    decision = router.classify(question)

    if decision:
        domain = decision.domain
        console.print(
            f"[dim]→ Routing to [bold]{domain}[/bold] expert "
            f"(confidence: {decision.confidence:.0%}) — {decision.reasoning}[/dim]\n"
        )
    else:
        domain = _show_fallback_menu()

    # ── Render the expert prompt ──────────────────────────────────────────────
    # code_reviewer needs 'code' and 'context' — handle separately
    if domain == "code_reviewer":
        code = Prompt.ask("Paste your Go code (single line or file path)")
        context = Prompt.ask("Context (what does this code do?)")
        rendered = library.render(domain, code=code, context=context)
    else:
        rendered = library.render(domain, question=question)

    # ── Stream the answer ─────────────────────────────────────────────────────
    client = LLMClient(config)
    output_chunks = []
    start = time.time()

    console.print(Panel(
        f"[bold]{domain.replace('_', ' ').title()} Expert[/bold]",
        border_style="cyan",
    ))

    for chunk in client.stream(
        rendered["user"],
        system=rendered["system"],
        override_temp=rendered["temperature"],
    ):
        print(chunk, end="", flush=True)
        output_chunks.append(chunk)

    latency = round(time.time() - start, 2)
    print()

    full = "".join(output_chunks)
    out_tokens = client.count_tokens(full)
    in_tokens = client.count_tokens(rendered["user"] + rendered["system"])
    cost = estimate_cost(in_tokens, out_tokens)

    console.print(
        f"\n[dim][Tokens: {in_tokens:,} in / {out_tokens:,} out | "
        f"Cost: ${cost:.4f} | Latency: {latency}s][/dim]\n"
    )
