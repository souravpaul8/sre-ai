# src/llm_playground_cli/commands/agent.py
"""
Interactive SRE agent with tool calling.

This is a multi-turn chat loop where:
1. User asks a question
2. The LLM decides which tools to call (or answers directly)
3. Tools execute and results feed back to the LLM
4. The LLM synthesizes a final answer
5. Conversation persists across turns (the agent has memory)
"""

import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import GenerationConfig
from llm_playground_cli.tools.registry import ToolRegistry
from llm_playground_cli.tools.primitives import ALL_BUILTIN_TOOLS
from llm_playground_cli.tools.k8s import ALL_K8S_TOOLS

console = Console()

AGENT_SYSTEM_PROMPT = """You are an SRE AI assistant with access to tools
for interacting with the local filesystem, a Kubernetes cluster, and an internal knowledge base.

## Available Capabilities
- Search the knowledge base / runbooks for policies, error codes, and troubleshooting steps
- Read files from disk (configs, logs, scripts)
- List directory contents
- Run shell commands (with user approval)
- Check Kubernetes pod status
- Describe Kubernetes deployments
- View Kubernetes pod logs

## Rules
1. ALWAYS use tools to get real data — never guess or make up file
   contents, pod status, or command output.
2. When answering questions about incident playbooks, runbooks, error codes, or company architecture,
   ALWAYS check `search_knowledge_base` first.
3. When diagnosing Kubernetes issues, start with get_pod_status,
   then get_logs if needed, then describe_deployment for config.
4. After getting tool results, explain them in plain language.
5. If a tool returns an error, explain what went wrong and suggest fixes.
6. For run_shell, prefer specific tools when available
   (use get_pod_status instead of run_shell with kubectl get pods).
7. Be concise but thorough. Use bullet points for findings.
"""


def _build_registry() -> ToolRegistry:
    """Build the tool registry with all available tools."""
    registry = ToolRegistry()
    for tool in ALL_BUILTIN_TOOLS:
        registry.register(tool)
    for tool in ALL_K8S_TOOLS:
        registry.register(tool)
    return registry


def _show_tools_table(registry: ToolRegistry) -> None:
    """Display available tools in a pretty table."""
    table = Table(title="Available Tools", border_style="cyan")
    table.add_column("Tool", style="bold cyan")
    table.add_column("Description", style="dim")
    table.add_column("⚠️", justify="center")

    for info in registry.list_tools():
        danger = "🔒" if info["dangerous"] else ""
        # Take just the first sentence of the description
        short_desc = info["description"].split(". ")[0] + "."
        table.add_row(info["name"], short_desc, danger)

    console.print(table)


def run(config: GenerationConfig) -> None:
    """Main entry point for `playground agent`."""
    console.print(
        Panel(
            "[bold cyan]SRE AI Agent[/bold cyan]\n"
            "[dim]Multi-turn assistant with tool calling. Type 'exit' to quit.[/dim]",
            border_style="cyan",
        )
    )

    registry = _build_registry()
    _show_tools_table(registry)

    client = LLMClient(config)
    all_tools = registry.get_all()

    # Callback: called each time the LLM invokes a tool
    def on_tool_call(name: str, args: dict) -> bool:
        tool = registry.get(name)
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())

        # If the tool is dangerous, ask for confirmation
        if tool and tool.is_dangerous:
            console.print(f"\n  [bold yellow]⚠️  {name}[/bold yellow]({args_str})")
            confirm = Prompt.ask(
                "  [yellow]Allow this command?[/yellow]",
                choices=["y", "n"],
                default="n",
            )
            if confirm.strip().lower() != "y":
                console.print("  [red]Denied by user.[/red]")
                return False
            console.print("  [green]Approved by user.[/green]")
            return True
        else:
            console.print(f"\n  [dim]🔧 {name}({args_str})[/dim]")
            return True

    console.print()

    # ── Interactive Loop ──────────────────────────────────────
    while True:
        try:
            question = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question.strip():
            continue
        if question.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        start = time.time()

        # Run the agentic loop
        answer, tool_calls = client.agent_loop(
            prompt=question,
            system=AGENT_SYSTEM_PROMPT,
            tools=all_tools,
            on_tool_call=on_tool_call,
        )

        latency = round(time.time() - start, 2)

        # Display the answer
        console.print(f"\n[bold cyan]Agent[/bold cyan]: {answer}")

        # Show tool call summary
        if tool_calls:
            calls_str = " → ".join(tc["name"] for tc in tool_calls)
            console.print(
                f"\n[dim]  Tools used: {calls_str} | Latency: {latency}s[/dim]"
            )
        else:
            console.print(f"\n[dim]  (no tools called) | Latency: {latency}s[/dim]")

        console.print()
