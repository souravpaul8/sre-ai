# LLM Playground CLI (`sre-ai`)

A powerful, modular command-line interface built to demonstrate core Large Language Model (LLM) integration concepts—including token counting, parameter tuning, structured outputs via Pydantic schemas, and real-time streaming responses—using the Google Gemini API.

---

## Features

- **Streaming Summarization:** Stream real-time summaries and insights for logs, text files, and source code directly to your terminal.
- **Structured File Analysis:** Parse and evaluate configuration files (such as Kubernetes manifests, Dockerfiles, and YAML configs) to output strict, validated Pydantic schemas detailing issues, recommendations, risk levels, and summaries.
- **Automated Documentation & Generation:** Scan project directories contextually to stream pristine README files or convert raw shell scripts into comprehensive, well-structured technical documentation.
- **Transparent Token & Cost Metrics:** Automatically calculate input/output token counts, model generation latency, and estimated API costs after executing any command.
- **Flexible Configuration:** Easily override generation parameters globally or on-the-fly via command-line flags (`--model`, `--temperature`, `--max-tokens`).

---

## Project Structure

```
sre-ai/
├── deployment.yaml
├── pyproject.toml
├── .env.example
├── sample.log
└── src/
    └── llm_playground_cli/
        ├── __init__.py
        ├── cli.py                  # CLI entry point (argparse command routing)
        ├── client.py               # GenAI Client wrapper (streaming, structured output)
        ├── config.py               # Generation configuration & environment loaders
        ├── commands/
        │   ├── __init__.py
        │   ├── convert.py          # Shell script -> structured documentation
        │   ├── explain.py          # Configuration file analysis
        │   ├── generate.py         # Automated README generation
        │   └── summarize.py        # Real-time streaming file summaries
        └── schemas/
            ├── __init__.py
            └── analysis.py         # Pydantic validation models
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sre-ai.git
   cd sre-ai
   ```

2. **Install the package in editable mode with dependencies:**
   ```bash
   pip install -e .
   ```

3. **Configure your environment variables:**
   Copy the template and add your Google Gemini API key (obtained from [Google AI Studio](https://aistudio.google.com)):
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API key:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   DEFAULT_MODEL=gemini-2.0-flash-lite
   DEFAULT_TEMPERATURE=0.3
   DEFAULT_MAX_OUTPUT_TOKENS=2048
   DEFAULT_TOP_P=0.95
   ```

---

## Usage Examples

The CLI exposes the `playground` entry point with global override flags and distinct subcommands.

### 1. Summarize a Log File (Streaming)
Stream an analysis of a log file directly to your terminal:
```bash
playground summarize sample.log --temperature 0.3
```

### 2. Explain a Configuration File (Structured Output)
Analyze a Kubernetes deployment manifest or configuration file against strict Pydantic schemas:
```bash
playground explain deployment.yaml
```

### 3. Generate a Project README
Scan your project directory context and stream out a comprehensive `README.md`:
```bash
playground generate readme --dir .
```

### 4. Convert Shell Scripts to Documentation
Parse a shell script and output structured tables detailing commands, flags, and categories:
```bash
playground convert script.sh
```

### Global Overrides
You can adjust generation parameters on the fly for any command:
```bash
playground summarize sample.log --model gemini-2.0-flash --temperature 0.7 --max-tokens 1024
```