from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import GenerationConfig
from llm_playground_cli.schemas.analysis import FileAnalysis, RiskLevel

console = Console()

RISK_COLORS = {
    RiskLevel.LOW: "green",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.HIGH: "red",
}

COST_PER_MILLION_INPUT = 0.10
COST_PER_MILLION_OUTPUT = 0.40

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT
    return round(input_cost + output_cost, 6)


def run(file_path: str, config: GenerationConfig,
 override_temp: float | None = None) -> None:

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        return
    content = path.read_text(encoding="utf-8")

    client = LLMClient(config)
    input_tokens = client.count_tokens(content)
    console.print(
        f"\n[bold cyan]Explaining[/bold cyan] [yellow]{path.name}[/yellow] "
        f"([green]{input_tokens:,}[/green] tokens)...\n"
    )

    system_prompt = (
        "You are an expert SRE and DevOps engineer. Analyze the provided file "
        "and return a structured assessment. Be specific about issues — "
        "don't be generic. Reference actual values from the file."
    )
    user_prompt = (
        f"Analyze this file ({path.name}) and return a structured JSON analysis:\n\n"
        f"{content}"
    )

    result, input_tokens, output_tokens, latency = client.generate_structured(
        prompt=user_prompt,
        schema=FileAnalysis,
        system=system_prompt,
        override_temp=override_temp,
    )

    risk_color = RISK_COLORS.get(result.risk_level, "white")
    # Summary panel
    console.print(Panel(
        f"[bold]{result.summary}[/bold]",
        title=f"[cyan]{result.file_type}[/cyan]",
        border_style=risk_color,
    ))

    if result.issues:
        issues_table = Table(
            title="⚠️  Issues Found",
            box=box.ROUNDED,
            border_style="yellow",
            show_header=False,
        )
        issues_table.add_column("Issue", style="yellow")
        for issue in result.issues:
            issues_table.add_row(f"• {issue}")
        console.print(issues_table)


    if result.recommendations:
        rec_table = Table(
            title="✅  Recommendations",
            box=box.ROUNDED,
            border_style="green",
            show_header=False,
        )
        rec_table.add_column("Recommendation", style="green")
        for rec in result.recommendations:
            rec_table.add_row(f"→ {rec}")
        console.print(rec_table)
    
    console.print(
        f"\nRisk Level: [{risk_color}][bold]{result.risk_level.value.upper()}[/bold][/{risk_color}]"
    )

    cost = estimate_cost(input_tokens, output_tokens)
    console.print(
        f"[dim][Tokens: {input_tokens:,} in / {output_tokens:,} out | "
        f"Cost: ${cost:.4f} | Latency: {latency}s][/dim]\n"
    )