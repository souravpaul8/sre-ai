## LLM Playground CLI

### Problem Statement

Build a command-line tool that demonstrates all Day 1 concepts: token counting, parameter tuning, structured output, and streaming.

### Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                   LLM Playground CLI             │
├──────────────────────────────────────────────────┤
│                                                  │
│  Commands:                                       │
│    summarize <file>      → Streaming summary     │
│    explain <file>        → Structured analysis   │
│    generate readme       → Streaming generation  │
│    convert <file>        → Structured output     │
│    tokens <file>         → Token analysis        │
│    cost <file>           → Cost estimation       │
│                                                  │
├──────────────────────────────────────────────────┤
│                  Core Modules                    │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Tokenizer│ │ LLM      │ │ Output           │ │
│  │ Module   │ │ Client   │ │ Formatter        │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│                                                  │
├──────────────────────────────────────────────────┤
│              Configuration                       │
│  ┌──────────────────────────────────────────────┐│
│  │ Model, Temperature, Max Tokens, Provider     ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  LLM API         │
│  (OpenAI /       │
│   Anthropic)     │
└──────────────────┘
```

### Folder Structure

```
llm-playground/
├── README.md
├── pyproject.toml
├── .env.example                # OPENAI_API_KEY=sk-...
├── src/
│   └── playground/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (argparse)
│       ├── client.py           # LLM client wrapper
│       ├── tokenizer.py        # Token counting, analysis
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── summarize.py    # Streaming summarization
│       │   ├── explain.py      # Structured output analysis
│       │   ├── generate.py     # README generation
│       │   └── convert.py      # Shell → docs conversion
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── analysis.py     # Pydantic models for structured output
│       └── config.py           # Generation configs
└── tests/
    ├── test_tokenizer.py
    ├── test_client.py
    └── test_commands.py
```

### Implementation Steps

1. **Set up project** — `pyproject.toml` with `openai`, `tiktoken`, `pydantic` dependencies
2. **Build tokenizer module** — Count tokens, show token breakdown, estimate cost
3. **Build LLM client** — Wrapper with streaming, structured output, error handling, logging
4. **Implement `summarize`** — Streaming output to terminal
5. **Implement `explain`** — Structured output with Pydantic schema
6. **Implement `generate`** — Streaming README generation
7. **Implement `convert`** — Structured output (shell commands → documented YAML/Markdown)
8. **Add token counting** — Show input/output tokens and cost after every command
9. **Add parameter flags** — `--temperature`, `--model`, `--max-tokens`

### Example Usage

```bash
# Summarize a log file (streaming)
$ playground summarize logs.txt --temperature 0.3
Analyzing logs.txt (4,231 tokens)...
The logs show a recurring OOMKilled error on the payment-service pod...
[Tokens: 4231 in / 287 out | Cost: $0.004 | Latency: 2.1s]

# Explain a deployment YAML (structured output)
$ playground explain deployment.yaml
{
  "summary": "nginx deployment with 3 replicas",
  "issues": ["No resource limits", "Using :latest tag"],
  "recommendations": ["Add resource requests/limits", "Pin image version"],
  "risk_level": "medium"
}
[Tokens: 1205 in / 156 out | Cost: $0.005 | Latency: 1.4s]

# Generate a README (streaming)
$ playground generate readme --model gpt-4o
# My Kubernetes Monitoring Stack
This project implements a comprehensive monitoring solution...
[Tokens: 892 in / 1243 out | Cost: $0.015 | Latency: 8.2s]

# Count tokens in a file
$ playground tokens deployment.yaml
File: deployment.yaml
Tokens: 1,205 (using cl100k_base)
Estimated cost (GPT-4o): $0.003 input
Estimated cost (GPT-4o-mini): $0.0002 input
```

### Production Improvements

- Add retry logic with exponential backoff
- Add support for multiple providers (OpenAI, Anthropic, Gemini)
- Add prompt caching for repeated system prompts
- Add cost tracking and budget alerts
- Add output caching (hash prompt → cached response)
- Add rate limiting for batch operations

### Scaling Strategy

- Use async for concurrent file processing
- Implement request queuing for large batches
- Use OpenAI Batch API for non-realtime bulk processing
- Cache results to avoid duplicate processing

### Monitoring

- Log every API call with tokens, cost, latency
- Track daily/weekly cost trends
- Alert on unusual cost spikes
- Monitor error rates and retry counts

### Testing Strategy

```python
# Test tokenizer accuracy
def test_token_count():
    count = count_tokens("Hello world", model="gpt-4o")
    assert count == 2

# Test structured output schema validation
def test_explain_schema():
    result = explain("test_deployment.yaml")
    assert isinstance(result, DeploymentAnalysis)
    assert result.risk_level in ["low", "medium", "high"]

# Test streaming collects all tokens
def test_streaming_complete():
    tokens = list(stream_response("Say hello"))
    full = "".join(tokens)
    assert len(full) > 0

# Test error handling
def test_retry_on_rate_limit():
    # Mock rate limit response, verify retry behavior
    ...
```
