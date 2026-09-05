import argparse
import sys
from dataclasses import replace

from llm_playground_cli.config import get_default_config
from llm_playground_cli.commands import summarize, explain, generate, convert, ask, agent, ingest


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

    # ── ingest ─────────────────────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a document (.pdf, .md, .txt) or text into the knowledge base",
    )
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument(
        "--file", "-f",
        help="Path to file to ingest (.pdf, .md, .txt)",
    )
    ingest_group.add_argument(
        "--text", "-t",
        help="Raw text snippet to ingest",
    )
    ingest_parser.add_argument(
        "--collection",
        default="sre_docs_v2",
        help="Collection name in Qdrant (default: sre_docs_v2)",
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
    
    elif args.command == "ingest":
        ingest.run(
            config=config,
            file_path=args.file,
            text=args.text,
            collection_name=args.collection
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()