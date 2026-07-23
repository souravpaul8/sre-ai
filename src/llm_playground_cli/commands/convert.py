from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import GenerationConfig
from llm_playground_cli.schemas.analysis import ShellDocumentation

console = Console()

COST_PER_MILLION_INPUT = 0.10
COST_PER_MILLION_OUTPUT = 0.40


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT
    return round(input_cost + output_cost, 6)


def run(file_path: str, config: GenerationConfig, override_temp: float | None = None) -> None:
    
    path = Path(file_path)

    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        return

    content = path.read_text(encoding="utf-8")
    client = LLMClient(config)
    input_tokens = client.count_tokens(content)

    console.print(
        f"\n[bold cyan]Converting[/bold cyan] [yellow]{path.name}[/yellow] "
        f"([green]{input_tokens:,}[/green] tokens) to documentation...\n"
    )

    system_prompt = (
        "You are a technical writer specializing in DevOps and shell scripting. "
        "Convert the provided shell commands into structured documentation. "
        "For each command, explain what it does, document its flags, and give a realistic example output."
    )

    user_prompt = (
        f"Document all the shell commands in this file ({path.name}):\n\n{content}"
    )

    result, input_tokens, output_tokens, latency = client.generate_structured(
        prompt=user_prompt,
        schema=ShellDocumentation,
        system=system_prompt,
        override_temp=override_temp,
    )

    console.print(Panel(
        f"[bold]{result.title}[/bold]",
        border_style="cyan",
        title="📖 Shell Documentation",
    ))

    if result.prerequisites:
        console.print("\n[bold yellow]Prerequisites:[/bold yellow]")
        for prereq in result.prerequisites:
            console.print(f"  • {prereq}")

    console.print()
    for i, cmd in enumerate(result.commands, 1):
        table = Table(
            title=f"[bold cyan]Command {i}:[/bold cyan] [white]{cmd.command}[/white]",
            box=box.ROUNDED,
            border_style="blue",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Field", style="cyan", width=18)
        table.add_column("Detail", style="white")

        table.add_row("Category", f"[magenta]{cmd.category}[/magenta]")
        table.add_row("Description", cmd.description)
        table.add_row("Flags", "\n".join(cmd.flags) if cmd.flags else "—")
        table.add_row("Example Output", f"[dim]{cmd.example_output}[/dim]")

        console.print(table)
        console.print()

    cost = estimate_cost(input_tokens, output_tokens)
    console.print(
        f"[dim][Tokens: {input_tokens:,} in / {output_tokens:,} out | "
        f"Cost: ${cost:.4f} | Latency: {latency}s][/dim]\n"
    )
