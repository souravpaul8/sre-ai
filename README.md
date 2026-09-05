# sre-buddy 🛠️

**`sre-buddy`** is your autonomous AI assistant and debugging companion for Site Reliability Engineering (SRE), Kubernetes operations, and incident troubleshooting. Powered by Google Gemini and equipped with a persistent local RAG knowledge base.

---

## 🚀 Quick Start & Installation

Install directly via `pip` or `pipx`:

```bash
# Install with pip
pip install sre-buddy

# Or run in an isolated environment with pipx (Recommended)
pipx install sre-buddy
```

*(From TestPyPI)*:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sre-buddy
```

Once installed, you can use either **`sre-buddy`** or **`playground`** commands in your terminal:

```bash
sre-buddy --help
```

---

## ⚙️ Environment Variables

Create a `.env` file in your working directory or export the variables in your shell profile (`~/.zshrc` or `~/.bashrc`).

### Required Variable

| Variable | Description |
| :--- | :--- |
| `GOOGLE_API_KEY` | **Required.** Your API key from [Google AI Studio](https://aistudio.google.com). |

### Optional Configurations & Defaults

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DEFAULT_MODEL` | `gemini-3.5-flash-lite` | Primary LLM model for analysis & agent execution. |
| `ROUTER_MODEL` | `gemini-3.5-flash-lite` | Model used for intent and question routing. |
| `DEFAULT_TEMPERATURE` | `0.3` | Generation randomness / temperature (0.0 to 1.0). |
| `DEFAULT_MAX_OUTPUT_TOKENS` | `4096` | Max token limit per LLM response turn. |
| `DEFAULT_TOP_P` | `0.9` | Top-p sampling probability. |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Google embedding model for RAG vector search. |
| `QDRANT_COLLECTION_NAME` | `sre_docs_v2` | Qdrant vector database collection name. |
| `QDRANT_VECTOR_SIZE` | `3072` | Embedding vector dimensions (3072 for gemini-embedding-2). |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder model for re-ranking search results. |
| `RERANKER_MAX_LENGTH` | `512` | Max token length for the cross-encoder re-ranker. |
| `AGENT_MAX_ITERATIONS` | `10` | Maximum autonomous tool execution steps before stopping. |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | Suppress Hugging Face download progress bars in CLI. |

### Example `.env` file
```env
GOOGLE_API_KEY=AIzaSy...
DEFAULT_MODEL=gemini-3.5-flash-lite
AGENT_MAX_ITERATIONS=10
```

---

## 📖 How to Run

### 1. Interactive Autonomous SRE Agent (`agent`)
Launch the multi-turn agent equipped with filesystem access, Kubernetes inspection tools, and knowledge base search:
```bash
sre-buddy agent
```
Features built into the agent:
- **Kubernetes Inspection**: Reads pod statuses, logs, and deployment specifications.
- **Human-in-the-Loop Confirmation**: Pauses and requests `[y/N]` confirmation before executing dangerous commands (like shell scripts).
- **Hybrid RAG Knowledge Retrieval**: Automatically searches your local playbooks with vector + BM25 keyword search.
- **Durable Checkpointing**: Saves state after every step. If you crash or press `Ctrl+C`, you are prompted to resume with zero lost context!
- **Trajectory Tracing**: Prints a structured telemetry table showing step-by-step tool latency and token consumption.

---

### 2. Ingest Runbooks & Documentation into Knowledge Base (`ingest`)
Feed markdown playbooks, incident documents, PDFs, or raw text into your local Qdrant vector database:

```bash
# Ingest raw text directly
sre-buddy ingest --text "When payments-service fails with error code 503, restart pod payments-worker-0 first."

# Ingest a markdown playbook or guide
sre-buddy ingest --file ./docs/incident-playbook.md

# Ingest a PDF runbook
sre-buddy ingest --file ./runbooks/architecture.pdf
```

---

### 3. Ask Technical Questions (`ask`)
Route questions to specialized prompt strategies automatically:
```bash
sre-buddy ask "How do I calculate MTTR and error budgets for high-throughput APIs?"
```

---

### 4. Analyze Configurations (`explain`)
Inspect Kubernetes YAMLs, Dockerfiles, or configs with validated Pydantic analysis (risk level, misconfigurations, recommendations):
```bash
sre-buddy explain deployment.yaml
```

---

### 5. Stream Log & File Summaries (`summarize`)
Stream real-time log summaries with token and latency metrics:
```bash
sre-buddy summarize sample.log --temperature 0.2
```

---

## 🛡️ Architecture & Security
- **Untrusted Observation Sandboxing**: External tool outputs and logs are encapsulated within strict `<tool_observation>` XML boundaries to prevent indirect prompt injection.
- **Cycle Circuit-Breaker**: Prevents infinite tool-calling loops by catching repetitive calls before token waste occurs.
- **Local Persistence**: Vector embeddings and state checkpoints are stored locally in your user home directory (`~/.sre-ai-rag.db` and `~/.sre-agent-state.json`).