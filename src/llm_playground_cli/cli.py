import argparse
import sys
from dataclasses import replace

from llm_playground_cli.config import get_default_config
from llm_playground_cli.commands import summarize, explain, generate, convert, ask, agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="playground",
        description="LLM Playground CLI — analyze files with Gemini",
    )

    # Global flags
    parser.add_argument(
        "--model",
        help="Override the default model (e.g. gemini-2.0-flash)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override generation temperature (0.0–1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override max output tokens",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── summarize ─────────────────────────────────────────────────────────────
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Stream a summary of a file",
    )
    summarize_parser.add_argument("file", help="Path to the file to summarize")

    # ── explain ───────────────────────────────────────────────────────────────
    explain_parser = subparsers.add_parser(
        "explain",
        help="Get a structured analysis of a YAML, Dockerfile, or config file",
    )
    explain_parser.add_argument("file", help="Path to the file to explain")

    # ── generate ──────────────────────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a README for the current project",
    )
    generate_parser.add_argument(
        "type",
        choices=["readme"],
        help="What to generate (only 'readme' supported for now)",
    )
    generate_parser.add_argument(
        "--dir",
        default=".",
        help="Project directory to scan (default: current directory)",
    )

    # ── convert ───────────────────────────────────────────────────────────────
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a shell script to structured documentation",
    )
    convert_parser.add_argument("file", help="Path to the shell script to document")

    # ── ask ─────────────────────────────────────────────────────────────────────
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a technical question — auto-routes to the right expert",
    )
    ask_parser.add_argument("question", nargs="?", help="The question to ask (optional)")

    # ── agent ─────────────────────────────────────────────────────────────────
    agent_parser = subparsers.add_parser(
        "agent",
        help="Interactive SRE agent with tool calling (reads files, runs kubectl, etc.)",
    )


    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = get_default_config()

    if args.model:
        config = replace(config, model=args.model)
    if args.max_tokens:
        config = replace(config, max_output_tokens=args.max_tokens)

    override_temp = args.temperature

    if args.command == "summarize":
        summarize.run(args.file, config, override_temp)

    elif args.command == "explain":
        explain.run(args.file, config, override_temp)

    elif args.command == "generate":
        generate.run(args.dir, config, override_temp)

    elif args.command == "convert":
        convert.run(args.file, config, override_temp)
    
    elif args.command == "ask":
        ask.run(config, args.question)
        
    elif args.command == "agent":
        agent.run(config)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()