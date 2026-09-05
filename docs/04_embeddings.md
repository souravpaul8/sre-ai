# Day 4 — Embeddings

## Goal

Understand how meaning becomes math.

Not the linear algebra.

Not the matrix operations.

The engineering.

An embedding is a **numerical representation of meaning**. It converts text — words, sentences, paragraphs, entire documents — into a list of numbers (a vector) such that **semantically similar texts have similar vectors**.

Think of it as coordinates representing meaning.

Just like GPS coordinates represent a physical location on Earth, embedding vectors represent a semantic location in meaning space. Two texts that mean similar things will have coordinates that are close together.

This is the foundation of every modern AI search system, every RAG pipeline, every recommendation engine, and every semantic matching feature in production AI applications.

---

## Big Picture

### Where Embeddings Fit in the AI Stack

```
┌─────────────────────────────────────────┐
│              User Interface             │
├─────────────────────────────────────────┤
│          Application Layer              │
│   (Your Code, Business Logic)           │
├─────────────────────────────────────────┤
│           AI Orchestration              │
│   (Prompt Assembly, Tool Routing,       │
│    Memory, RAG, Agent Loops)            │
├─────────────────────────────────────────┤
│         Retrieval Layer                 │  ← You are here (Day 4)
│   (Embeddings, Vector Search,           │
│    Semantic Matching)                   │
├─────────────────────────────────────────┤
│            LLM API Layer                │
│   (Tokens, Parameters, Streaming,       │
│    Structured Output)                   │
├─────────────────────────────────────────┤
│          Model Infrastructure           │
│   (GPU Clusters, Inference Engines)     │
├─────────────────────────────────────────┤
│         Foundation Model                │
│   (Weights, Architecture, Training)     │
└─────────────────────────────────────────┘
```

Embeddings sit at the **retrieval layer** — below orchestration, above the raw LLM API. They are the mechanism that allows your AI system to **find relevant information** before sending it to the LLM.

### Why Frontier AI Companies Care

Every knowledge-intensive AI product depends on embeddings:

- **Perplexity** — embeds web pages, searches semantically to find relevant sources for answers
- **ChatGPT** — uses embeddings in its retrieval pipeline for file search in GPTs and Projects
- **Cursor** — embeds your entire codebase for semantic code search
- **GitHub Copilot** — embeds repository context for relevant code suggestions
- **Notion AI** — embeds your workspace for semantic Q&A over your notes

Without embeddings, these products cannot work. You cannot stuff an entire codebase or knowledge base into a single LLM context window. Embeddings solve the retrieval problem — finding the right information to put in the context window.

If you do not understand embeddings deeply, you cannot:

- Build RAG systems
- Implement semantic search
- Design recommendation features
- Debug retrieval quality issues
- Optimize vector search performance
- Pass an AI engineering interview

---

## Mental Model

### The Software Engineer's Model of Embeddings

Think of embeddings as a **hash function for meaning**.

```
Traditional hash:   hash("hello") → 0x5d41402abc4b
                    hash("hi")    → 0x49f68a5c8493

Embedding:          embed("hello") → [0.12, -0.34, 0.56, ..., 0.78]  (1536 dims)
                    embed("hi")    → [0.11, -0.33, 0.55, ..., 0.77]  (1536 dims)
```

But unlike hash functions, embeddings preserve **semantic relationships**:

| Property       | Hash Function                | Embedding Function                  |
| -------------- | ---------------------------- | ----------------------------------- |
| Input          | Any data                     | Text (or images, audio)             |
| Output         | Fixed-length bit string      | Fixed-length float vector           |
| Similar inputs | Completely different outputs | **Similar outputs** (close vectors) |
| Distance       | Meaningless                  | **Meaningful** (semantic distance)  |
| Reversible     | No                           | No                                  |
| Deterministic  | Yes                          | Yes (for same model version)        |

### The Coordinate Analogy

Imagine a three-dimensional space where:

- X-axis = technical depth (low → high)
- Y-axis = topic area (infrastructure → application)
- Z-axis = sentiment (negative → positive)

```
                        ▲ Application
                        │
                        │   • "React hooks tutorial"
                        │   • "How to build a REST API"
                        │
                        │
    Low technical ──────┼──────────── High technical ▶
                        │
                        │   • "CrashLoopBackOff in pod"
                        │   • "Kubernetes OOMKilled"
                        │   • "kubectl debug commands"
                        │
                        ▼ Infrastructure
```

In this simplified 3D space, "CrashLoopBackOff in pod" and "Kubernetes OOMKilled" would have coordinates that are close together — they are both about infrastructure issues with negative sentiment and medium-high technical depth.

Real embeddings work exactly like this, except with **1,536 dimensions** (or 256, 768, 3,072 depending on the model) instead of 3. Each dimension captures some aspect of meaning — topic, tone, specificity, domain, intent — learned automatically during training.

### The Distance = Difference in Meaning

```
embed("Kubernetes pod crash")    ←→  embed("Container keeps restarting")
                                      Distance: 0.15 (very similar)

embed("Kubernetes pod crash")    ←→  embed("How to bake a cake")
                                      Distance: 0.95 (very different)
```

Small distance = similar meaning.

Large distance = different meaning.

This is the entire foundation of semantic search.

---

## Internal Architecture

### How an Embedding Model Works

```
Input Text: "Why is my pod in CrashLoopBackOff?"
    │
    ▼
┌────────────────────────────────────────┐
│  Tokenizer                             │
│                                        │
│  "Why" "is" "my" "pod" "in"           │
│  "Crash" "Loop" "Back" "Off" "?"      │
│                                        │
│  → [3400, 318, 616, 24093, 287,       │
│     34326, 39516, 7282, 9362, 30]     │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Transformer Encoder                   │
│                                        │
│  Self-attention across all tokens      │
│  Multiple layers (12-24)               │
│  Each token gets a contextual          │
│  representation                        │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Pooling Layer                         │
│                                        │
│  Combine all token representations     │
│  into a single vector.                 │
│                                        │
│  Methods:                              │
│  - Mean pooling (average all tokens)   │
│  - CLS token (first token only)        │
│  - Last token pooling                  │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Normalization                         │
│                                        │
│  Normalize to unit length              │
│  (L2 normalization)                    │
│  so cosine similarity = dot product    │
└────────────────────┬───────────────────┘
                     │
                     ▼
Output: [0.023, -0.041, 0.089, ..., 0.017]
        ← 1536 floating point numbers →
```

Key insight: unlike LLMs that generate text token by token, embedding models process the **entire input at once** and produce a **single fixed-length vector**. There is no auto-regressive generation — just one forward pass.

### Embedding vs. LLM — The Critical Distinction

| Property     | Embedding Model                         | LLM (Generative)                     |
| ------------ | --------------------------------------- | ------------------------------------ |
| Input        | Text                                    | Text                                 |
| Output       | Fixed-length vector (e.g., 1536 floats) | Variable-length text (tokens)        |
| Process      | Single forward pass                     | Auto-regressive (token by token)     |
| Speed        | Very fast (~10-50ms)                    | Slower (~200-2000ms+)                |
| Cost         | Very cheap (~$0.02 per 1M tokens)       | Expensive (~$2.50-$15 per 1M tokens) |
| Purpose      | Represent meaning as numbers            | Generate new text                    |
| Statefulness | Stateless                               | Stateless                            |

> [!TIP]
> Embedding API calls are **10-100x cheaper** and **5-20x faster** than LLM completion API calls. This is why we use embeddings for retrieval (searching millions of documents) and LLMs only for the final generation step.

---

## Similarity Search

### What Is Similarity Search?

Similarity search (also called nearest neighbor search) is finding the items in a collection that are most similar to a query. With embeddings, this means finding the vectors closest to the query vector.

```
Query:  "Why CrashLoopBackOff?"
        ↓ embed
        → [0.23, -0.41, 0.89, ...]

Database:
  doc1: "Pod restart policy"           → [0.21, -0.39, 0.87, ...]  distance: 0.05 ✅
  doc2: "CrashLoopBackOff debugging"   → [0.22, -0.40, 0.88, ...]  distance: 0.03 ✅✅
  doc3: "How to bake bread"            → [0.91, 0.32, -0.14, ...]  distance: 0.95 ✗
  doc4: "Container exit codes"         → [0.25, -0.38, 0.82, ...]  distance: 0.10 ✅

Results (sorted by distance):
  1. doc2: "CrashLoopBackOff debugging"   (most similar)
  2. doc1: "Pod restart policy"
  3. doc4: "Container exit codes"
```

### The Two-Phase Pattern

Every embedding-based search system follows this pattern:

```
Phase 1 — Index Time (offline, once):

  Documents
      │
      ▼
  ┌──────────────────┐
  │  Chunk documents  │
  │  (split into      │
  │   ~500 token      │
  │   pieces)         │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Embed each chunk │
  │  (API call to     │
  │   embedding model)│
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Store vectors    │
  │  in vector DB     │
  │  (+ original text)│
  └──────────────────┘

Phase 2 — Query Time (online, per request):

  User Query
      │
      ▼
  ┌──────────────────┐
  │  Embed query      │
  │  (same model as   │
  │   indexing)        │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Search vector DB │
  │  (find top-K      │
  │   nearest vectors)│
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Return matching  │
  │  document chunks  │
  │  (+ similarity    │
  │   scores)         │
  └──────────────────┘
```

> [!IMPORTANT]
> **You must use the same embedding model for indexing and querying.** Vectors from different models are incompatible — they live in different vector spaces. Switching embedding models requires re-embedding your entire document collection.

---

## Cosine Similarity

### What Is It?

Cosine similarity measures how similar two vectors are by computing the cosine of the angle between them.

```
Cosine Similarity = (A · B) / (‖A‖ × ‖B‖)

Where:
  A · B     = dot product (sum of element-wise products)
  ‖A‖, ‖B‖ = magnitudes (lengths) of vectors A and B
```

### The Intuition — No Math Required

Think of two arrows pointing from the origin:

```
        ▲
       /
      /  ← Vector A
     /
    / θ  (small angle = similar)
   /─────────────▶ Vector B
  O

  cos(0°) = 1.0   → identical direction  → identical meaning
  cos(90°) = 0.0  → perpendicular        → unrelated
  cos(180°) = -1.0 → opposite direction  → opposite meaning
```

The angle between the arrows tells you how similar the meanings are:

| Cosine Similarity | Angle   | Interpretation              |
| ----------------- | ------- | --------------------------- |
| 1.0               | 0°      | Identical meaning           |
| 0.9 – 1.0         | ~26°    | Very similar                |
| 0.7 – 0.9         | ~37-45° | Related                     |
| 0.5 – 0.7         | ~45-60° | Somewhat related            |
| 0.0 – 0.5         | ~60-90° | Different topics            |
| < 0.0             | >90°    | Opposite / contradictory    |

### Why Cosine Similarity and Not Euclidean Distance?

| Metric             | Formula                | Sensitivity to Magnitude | Best For              |
| ------------------ | ---------------------- | ------------------------ | --------------------- |
| Cosine Similarity  | cos(θ) between vectors | No — direction only      | Text similarity       |
| Euclidean Distance | √(Σ(aᵢ - bᵢ)²)         | Yes — magnitude matters  | Spatial distances     |
| Dot Product        | Σ(aᵢ × bᵢ)             | Yes — magnitude matters  | Normalized embeddings |

Cosine similarity ignores vector magnitude (length) and only cares about direction. This is important because:

- A short document and a long document about the same topic should be similar
- The magnitude of an embedding vector reflects text length, not meaning
- Direction captures semantic content, magnitude captures superficial properties

> [!TIP]
> Most embedding APIs return **normalized vectors** (unit length). When vectors are normalized, cosine similarity equals the dot product. This is a massive performance optimization — dot products are much faster to compute than full cosine similarity.

### Python Implementation

```python
import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)

    dot_product = np.dot(a_np, b_np)
    magnitude_a = np.linalg.norm(a_np)
    magnitude_b = np.linalg.norm(b_np)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def cosine_similarity_normalized(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity for normalized vectors (unit length).
    When vectors are normalized, cosine similarity = dot product.
    Much faster — no need to compute magnitudes.
    """
    return np.dot(a, b)
```

### TypeScript Implementation

```typescript
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same dimensions");
  }

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magnitudeA += a[i] * a[i];
    magnitudeB += b[i] * b[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return dotProduct / (magnitudeA * magnitudeB);
}
```

---

## Semantic Search

### What Is Semantic Search?

Semantic search is finding documents based on **meaning**, not keywords.

```
Traditional keyword search:
  Query: "pod keeps dying"
  Result: No matches (no document contains exactly "pod keeps dying")

Semantic search:
  Query: "pod keeps dying"
  Result: "Troubleshooting CrashLoopBackOff in Kubernetes"
          (similar meaning, different words)
```

This is the killer feature of embeddings. Semantic search understands:

- **Synonyms**: "dying" ≈ "crashing" ≈ "failing" ≈ "CrashLoopBackOff"
- **Paraphrasing**: "pod keeps dying" ≈ "container restart loop"
- **Intent**: "why won't my pod start" ≈ "troubleshoot pod pending state"
- **Concepts**: "high memory usage" ≈ "OOMKilled"

### Keyword Search vs. Semantic Search

| Aspect         | Keyword Search (BM25/TF-IDF)     | Semantic Search (Embeddings)         |
| -------------- | -------------------------------- | ------------------------------------ |
| Matching       | Exact or stemmed word matching   | Meaning-based matching               |
| Synonyms       | Misses unless explicitly mapped  | Handles naturally                    |
| Typos          | Fails or requires fuzzy matching | Often handles minor typos            |
| Cross-language | Cannot match across languages    | Can match (multilingual models)      |
| Long queries   | Diluted signal, poor relevance   | Handles well                         |
| Rare terms     | Excellent (high IDF score)       | May miss very domain-specific terms  |
| Speed          | Very fast (inverted index)       | Slower (vector comparison)           |
| Infrastructure | Simple (Elasticsearch, Postgres) | Requires vector database             |
| Cost           | CPU only                         | Embedding API cost + GPU for vectors |

### Hybrid Search — The Production Approach

In production, **no one uses purely semantic search**. The best systems use **hybrid search** — combining keyword and semantic search:

```
Query: "CrashLoopBackOff exit code 137"
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
┌────────────────┐          ┌────────────────┐
│ Keyword Search │          │ Semantic Search │
│ (BM25)         │          │ (Embeddings)    │
│                │          │                 │
│ Matches exact  │          │ Matches meaning │
│ "CrashLoop     │          │ of the query    │
│  BackOff"      │          │                 │
│ and "137"      │          │                 │
└───────┬────────┘          └───────┬─────────┘
        │                           │
        └─────────┬─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Reciprocal Rank │
        │  Fusion (RRF)    │
        │                  │
        │  Combine and     │
        │  re-rank results │
        └────────┬────────┘
                 │
                 ▼
        Final ranked results
```

Hybrid search captures:

- **Exact matches** — the keyword "CrashLoopBackOff" appears in the document (keyword search excels here)
- **Semantic matches** — documents about container restart loops that don't use the exact term (semantic search excels here)
- **Specific values** — "exit code 137" is a precise value that keyword search handles better than semantic search

> [!IMPORTANT]
> **Hybrid search outperforms both pure keyword and pure semantic search in virtually all benchmarks.** This is an established best practice, not an opinion. If you are building a production retrieval system, implement hybrid search.

---

## Embedding Models — Comparison

### Available Embedding Models (2025)

| Model                         | Dimensions | Max Tokens | Cost (per 1M tokens) | Provider  | Notes                               |
| ----------------------------- | ---------- | ---------- | -------------------- | --------- | ----------------------------------- |
| `text-embedding-3-small`      | 1536       | 8191       | $0.02                | OpenAI    | Best cost/quality ratio             |
| `text-embedding-3-large`      | 3072       | 8191       | $0.13                | OpenAI    | Highest quality from OpenAI         |
| `voyage-3-large`              | 1024       | 32000      | $0.18                | Voyage AI | Excellent for code embeddings       |
| `embed-v4.0`                  | 1024       | 128000     | $0.10                | Cohere    | Very long context support           |
| Gemini embedding              | 768        | 2048       | $0.00                | Google    | Free tier available                 |
| `nomic-embed-text`            | 768        | 8192       | Free (self-hosted)   | Nomic     | Open source, self-hostable          |
| `bge-large-en-v1.5`          | 1024       | 512        | Free (self-hosted)   | BAAI      | Open source, widely used            |

### Choosing an Embedding Model

```
What is your priority?
    │
    ├── Cost → text-embedding-3-small (OpenAI)
    │          or nomic-embed-text (self-hosted, free)
    │
    ├── Quality → text-embedding-3-large (OpenAI)
    │             or voyage-3-large (Voyage AI)
    │
    ├── Code search → voyage-3-large (optimized for code)
    │
    ├── Long documents → embed-v4.0 (Cohere, 128K tokens)
    │
    ├── Self-hosted / privacy → nomic-embed-text or bge-large
    │
    └── Free tier → Gemini embedding
```

### Dimension Reduction

OpenAI's `text-embedding-3-*` models support **dimension reduction** — you can request a smaller vector without a separate model:

```python
# Full dimensions (3072)
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hello world",
)

# Reduced dimensions (256) — saves storage and search latency
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hello world",
    dimensions=256,
)
```

This is a powerful cost optimization lever:

| Dimensions | Storage per Vector | Quality Impact | Use Case                    |
| ---------- | ------------------ | -------------- | --------------------------- |
| 3072       | 12 KB              | Baseline       | Highest quality required    |
| 1536       | 6 KB               | Minimal loss   | Most production use cases   |
| 512        | 2 KB               | Some loss      | Large-scale, cost-sensitive |
| 256        | 1 KB               | Noticeable     | Very large scale, coarse filtering |

---

## Internal Mechanics

### How Embedding APIs Work

```python
from openai import OpenAI

client = OpenAI()


def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate an embedding vector for a text string."""
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def embed_batch(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in batches.

    The API accepts up to 2048 inputs per call.
    For large collections, batch to manage memory and rate limits.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch,
        )

        # Embeddings come back in the same order as inputs
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

### Anthropic/Voyage AI Embeddings

```python
# Voyage AI (recommended by Anthropic for embeddings)
import voyageai

voyage_client = voyageai.Client()


def embed_text_voyage(text: str) -> list[float]:
    result = voyage_client.embed(
        [text],
        model="voyage-3-large",
        input_type="query",  # or "document" for indexing
    )
    return result.embeddings[0]


def embed_documents_voyage(texts: list[str]) -> list[list[float]]:
    result = voyage_client.embed(
        texts,
        model="voyage-3-large",
        input_type="document",  # Use "document" for indexing
    )
    return result.embeddings
```

> [!TIP]
> Some embedding APIs (Voyage AI, Cohere) support separate `input_type` for queries vs. documents. Use `"query"` for search queries and `"document"` for indexed content. This optimizes the embedding for each use case — queries tend to be short questions, documents tend to be longer passages.

### TypeScript Example

```typescript
import OpenAI from "openai";

const client = new OpenAI();

async function embedText(text: string): Promise<number[]> {
  const response = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}

async function embedBatch(texts: string[]): Promise<number[][]> {
  const response = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: texts,
  });
  return response.data.map((item) => item.embedding);
}
```

---

## Performance Perspective

### Embedding Latency

| Operation                    | Typical Latency | Notes                           |
| ---------------------------- | --------------- | ------------------------------- |
| Single text embedding (API)  | 20–80ms         | Dominated by network round trip |
| Batch embedding (100 texts)  | 100–500ms       | Much more efficient than single |
| Vector search (10K vectors)  | <1ms            | In-memory, brute force          |
| Vector search (1M vectors)   | 1–10ms          | ANN index (HNSW)                |
| Vector search (100M vectors) | 5–50ms          | Distributed vector database     |

### Throughput Optimization

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()


async def embed_batch_async(
    texts: list[str],
    model: str = "text-embedding-3-small",
    max_concurrent: int = 5,
) -> list[list[float]]:
    """
    Embed texts with concurrent API calls for maximum throughput.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(texts)

    async def embed_chunk(index: int, chunk: list[str]):
        async with semaphore:
            response = await async_client.embeddings.create(
                model=model,
                input=chunk,
            )
            return index, [item.embedding for item in response.data]

    # Split into batches of 100
    batch_size = 100
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tasks.append(embed_chunk(i, batch))

    # Execute concurrently
    for coro in asyncio.as_completed(tasks):
        index, embeddings = await coro
        for j, emb in enumerate(embeddings):
            results[index + j] = emb

    return results
```

### Benchmarking Embedding Performance

| Metric                  | What to Measure                            | Why It Matters                      |
| ----------------------- | ------------------------------------------ | ----------------------------------- |
| Embedding latency (p50) | Time to embed a single query               | Affects search response time        |
| Batch throughput        | Texts embedded per second                  | Affects indexing speed              |
| Vector search latency   | Time to find top-K results                 | Affects query response time         |
| Recall@K                | Fraction of relevant docs in top K results | Measures search quality             |
| Index build time        | Time to build the vector index             | Affects deployment and update speed |
| Storage per vector      | Bytes per embedding vector                 | Affects infrastructure cost         |

---

## Cost Perspective

### Embedding Cost Model

```
Embedding cost = (total_tokens / 1,000,000) × price_per_million_tokens

Example:
  10,000 documents × 500 tokens/doc = 5,000,000 tokens
  Using text-embedding-3-small @ $0.02/1M tokens
  Cost = 5 × $0.02 = $0.10

  That is 10 cents to embed 10,000 documents.
```

### Cost Comparison

| Scenario                       | Documents  | Avg Tokens | Total Tokens | Model                     | Cost     |
| ------------------------------ | ---------- | ---------- | ------------ | ------------------------- | -------- |
| Small knowledge base           | 1,000      | 500        | 500K         | text-embedding-3-small    | $0.01    |
| Medium documentation           | 10,000     | 500        | 5M           | text-embedding-3-small    | $0.10    |
| Large codebase                 | 100,000    | 300        | 30M          | text-embedding-3-small    | $0.60    |
| Enterprise knowledge base      | 1,000,000  | 500        | 500M         | text-embedding-3-small    | $10.00   |
| Query embeddings (per day)     | 50,000     | 30         | 1.5M         | text-embedding-3-small    | $0.03    |

> [!TIP]
> Embedding costs are almost negligible compared to LLM generation costs. The expensive part of a RAG system is the **LLM generation step** (processing the retrieved context), not the embedding step. Optimize LLM costs first.

### Storage Cost

| Vector Count | Dimensions | Storage (Float32) | Storage (Float16) |
| ------------ | ---------- | ----------------- | ----------------- |
| 10,000       | 1536       | 58 MB             | 29 MB             |
| 100,000      | 1536       | 580 MB            | 290 MB            |
| 1,000,000    | 1536       | 5.8 GB            | 2.9 GB            |
| 10,000,000   | 1536       | 58 GB             | 29 GB             |

```
Storage per vector = dimensions × bytes_per_float
                   = 1536 × 4 bytes (float32)
                   = 6,144 bytes ≈ 6 KB
```

---

# Building a Semantic Search Engine

## The Complete Implementation

### Step 1 — Document Chunking

```python
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk of a document with its embedding."""
    id: str
    source: str        # Original document path or identifier
    content: str       # The text content of this chunk
    metadata: dict     # Additional metadata (page, section, etc.)
    embedding: list[float] | None = None


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    source: str = "",
) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks.

    Chunk size is in characters (not tokens).
    Overlap ensures context is preserved across chunk boundaries.
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 2
            else:
                # Look for sentence break
                period_pos = text.rfind(". ", start, end)
                if period_pos > start + chunk_size // 2:
                    end = period_pos + 2

        chunk_content = text[start:end].strip()

        if chunk_content:
            chunks.append(DocumentChunk(
                id=f"{source}:chunk_{chunk_index}",
                source=source,
                content=chunk_content,
                metadata={
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                },
            ))
            chunk_index += 1

        # Move forward, accounting for overlap
        start = end - chunk_overlap if end < len(text) else end

    return chunks
```

### Step 2 — Embedding and Indexing

```python
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

client = OpenAI()


class SemanticSearchEngine:
    """
    A simple but functional semantic search engine.

    Uses OpenAI embeddings and brute-force cosine similarity.
    Production systems would use a vector database (Pinecone,
    Qdrant, pgvector) instead of in-memory brute force.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, documents: list[dict[str, str]]) -> None:
        """
        Add documents to the search index.

        Each document should have 'content' and 'source' keys.
        Documents are chunked, embedded, and stored.
        """
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(
                text=doc["content"],
                source=doc.get("source", "unknown"),
                chunk_size=500,
                chunk_overlap=100,
            )
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Embed all chunks in batches
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embed_batch(texts)

        # Store
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        self.chunks.extend(all_chunks)

        # Build the numpy matrix for fast search
        self.embeddings = np.array(
            [chunk.embedding for chunk in self.chunks]
        )

        print(f"Index now contains {len(self.chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Search for chunks similar to the query.

        Returns top_k results sorted by similarity score.
        """
        if not self.chunks:
            return []

        # Embed the query
        query_embedding = self._embed_single(query)
        query_vector = np.array(query_embedding)

        # Compute cosine similarity with all chunks
        # For normalized vectors: cosine_sim = dot product
        similarities = np.dot(self.embeddings, query_vector)

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < similarity_threshold:
                continue

            results.append({
                "content": self.chunks[idx].content,
                "source": self.chunks[idx].source,
                "score": score,
                "chunk_id": self.chunks[idx].id,
                "metadata": self.chunks[idx].metadata,
            })

        return results

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        response = client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def _embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed multiple texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [
                item.embedding for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        return all_embeddings

    def save(self, path: str) -> None:
        """Save the index to disk."""
        data = {
            "model": self.model,
            "chunks": [
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding,
                }
                for chunk in self.chunks
            ],
        }
        Path(path).write_text(json.dumps(data))
        print(f"Index saved to {path} ({len(self.chunks)} chunks)")

    def load(self, path: str) -> None:
        """Load the index from disk."""
        data = json.loads(Path(path).read_text())
        self.model = data["model"]
        self.chunks = [
            DocumentChunk(
                id=c["id"],
                source=c["source"],
                content=c["content"],
                metadata=c["metadata"],
                embedding=c["embedding"],
            )
            for c in data["chunks"]
        ]
        self.embeddings = np.array(
            [chunk.embedding for chunk in self.chunks]
        )
        print(f"Index loaded from {path} ({len(self.chunks)} chunks)")
```

### Step 3 — Loading Kubernetes Documentation

```python
# Example: Loading Kubernetes troubleshooting documentation

K8S_DOCS = [
    {
        "source": "k8s-troubleshooting/crashloopbackoff",
        "content": """CrashLoopBackOff in Kubernetes

CrashLoopBackOff is a status indicating that a container in a pod is
repeatedly crashing and Kubernetes is backing off on restarting it.

The backoff timer starts at 10 seconds and doubles each time, up to a
maximum of 5 minutes: 10s, 20s, 40s, 80s, 160s, 300s.

Common causes:
1. Application error — the container exits with a non-zero exit code.
   Check the exit code: exit code 1 means general error, exit code 137
   means OOMKilled (out of memory), exit code 139 means segfault.
2. Missing dependencies — the application cannot connect to a required
   service (database, message queue, etc.) and crashes on startup.
3. Misconfigured command or args — the container's entrypoint or
   command is wrong.
4. Failing health checks — liveness probes fail, causing Kubernetes
   to restart the container.
5. Resource limits too low — the container is OOMKilled because memory
   limits are set too low for the application.

Debugging steps:
- kubectl describe pod <pod-name> — check Events section
- kubectl logs <pod-name> — check application logs
- kubectl logs <pod-name> --previous — check logs from crashed container
- kubectl get pod <pod-name> -o yaml — check exit code and reason
""",
    },
    {
        "source": "k8s-troubleshooting/oomkilled",
        "content": """OOMKilled in Kubernetes

OOMKilled means a container was terminated because it exceeded its
memory limit. The exit code is 137.

This happens when:
1. The container's memory usage exceeds the memory limit set in the
   pod spec (spec.containers[].resources.limits.memory).
2. The node runs out of memory and the kubelet evicts pods.

How to fix:
- Increase the memory limit in the pod spec.
- Profile the application to find memory leaks.
- Use JVM flags like -Xmx for Java applications.
- Check for memory leaks in Go with pprof.
- Implement proper resource requests AND limits.

The difference between requests and limits:
- requests: guaranteed minimum resources. Used for scheduling.
- limits: maximum resources. Container is killed if exceeded.

Best practice: set requests = limits for predictable behavior
(QoS class: Guaranteed). Or set requests < limits for burstable
workloads (QoS class: Burstable).
""",
    },
    {
        "source": "k8s-troubleshooting/pending-pods",
        "content": """Pods Stuck in Pending State

A pod in Pending state has been accepted by the Kubernetes cluster
but is not yet running. Common causes:

1. Insufficient resources — no node has enough CPU or memory to
   schedule the pod. Check: kubectl describe pod <name> for events
   showing "Insufficient cpu" or "Insufficient memory".

2. Node selectors or affinity rules — the pod requires specific
   nodes (nodeSelector, nodeAffinity) and none are available.

3. Taints and tolerations — all suitable nodes have taints that
   the pod does not tolerate.

4. PersistentVolumeClaim pending — the pod requires a PVC that
   hasn't been bound to a PV yet.

5. Image pull issues — the container image cannot be pulled
   (ImagePullBackOff is a separate status, but can cause Pending).

6. ResourceQuota exceeded — the namespace has a ResourceQuota
   and the pod would exceed it.

Debugging:
- kubectl describe pod <name> — check the Events section
- kubectl get nodes — check node status
- kubectl describe node <name> — check allocatable vs allocated
- kubectl get pvc — check PVC status
""",
    },
    {
        "source": "k8s-networking/service-types",
        "content": """Kubernetes Service Types

Services expose pods to network traffic. Four types:

1. ClusterIP (default) — Internal cluster IP. Only accessible
   within the cluster. Use for internal microservice communication.

2. NodePort — Exposes on each node's IP at a static port (30000-32767).
   Accessible from outside the cluster via <NodeIP>:<NodePort>.
   Not recommended for production.

3. LoadBalancer — Provisions an external load balancer (cloud provider).
   The standard way to expose services to the internet.

4. ExternalName — Maps a service to a DNS name (CNAME record).
   No proxying. Used for pointing to external services.

Choosing the right type:
- Internal only → ClusterIP
- External traffic → LoadBalancer (cloud) or Ingress
- Development/testing → NodePort
- External DNS alias → ExternalName
""",
    },
    {
        "source": "k8s-troubleshooting/imagepullbackoff",
        "content": """ImagePullBackOff in Kubernetes

ImagePullBackOff means Kubernetes cannot pull the container image.

Common causes:
1. Image does not exist — wrong image name or tag.
2. Private registry — missing imagePullSecrets.
3. Network issues — node cannot reach the registry.
4. Rate limiting — Docker Hub rate limits (100 pulls/6hrs for
   anonymous, 200 for free accounts).

Debugging:
- kubectl describe pod <name> — check Events for pull errors
- kubectl get pod <name> -o yaml — check image name
- Try pulling manually: docker pull <image> on the node

Fix:
- Verify image name and tag exist in the registry
- Add imagePullSecrets to the pod spec or service account
- Use a private registry mirror to avoid rate limits
- Check node network connectivity to the registry
""",
    },
]


# Initialize and populate the search engine
engine = SemanticSearchEngine(model="text-embedding-3-small")
engine.add_documents(K8S_DOCS)
engine.save("k8s_search_index.json")
```

### Step 4 — Searching

```python
# Search for relevant documentation
results = engine.search("Why CrashLoopBackOff?", top_k=3)

for result in results:
    print(f"\n{'='*60}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Content: {result['content'][:200]}...")
```

**Expected output:**

```
============================================================
Score: 0.8734
Source: k8s-troubleshooting/crashloopbackoff
Content: CrashLoopBackOff in Kubernetes

CrashLoopBackOff is a status indicating that a container in a pod is
repeatedly crashing and Kubernetes is backing off on restarting it...

============================================================
Score: 0.7921
Source: k8s-troubleshooting/oomkilled
Content: OOMKilled in Kubernetes

OOMKilled means a container was terminated because it exceeded its
memory limit. The exit code is 137...

============================================================
Score: 0.6543
Source: k8s-troubleshooting/imagepullbackoff
Content: ImagePullBackOff in Kubernetes

ImagePullBackOff means Kubernetes cannot pull the container image...
```

### Step 5 — Complete CLI

```python
#!/usr/bin/env python3
"""
Kubernetes Documentation Semantic Search CLI.

Usage:
    python search_cli.py "Why CrashLoopBackOff?"
    python search_cli.py "How do I expose a service externally?"
    python search_cli.py --index docs/    # Re-index documentation
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over Kubernetes documentation"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query in natural language",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Minimum similarity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--index",
        type=str,
        help="Path to a directory of Markdown docs to index",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default="k8s_search_index.json",
        help="Path to the saved index file",
    )
    args = parser.parse_args()

    engine = SemanticSearchEngine()

    if args.index:
        # Index a directory of Markdown files
        docs_dir = Path(args.index)
        documents = []
        for md_file in docs_dir.glob("**/*.md"):
            content = md_file.read_text()
            documents.append({
                "source": str(md_file),
                "content": content,
            })
        print(f"Indexing {len(documents)} documents from {docs_dir}...")
        engine.add_documents(documents)
        engine.save(args.index_file)
        print("Indexing complete.")
        return

    if not args.query:
        parser.print_help()
        return

    # Load the index and search
    try:
        engine.load(args.index_file)
    except FileNotFoundError:
        print(f"Index file not found: {args.index_file}")
        print("Run with --index <docs_dir> first to build the index.")
        sys.exit(1)

    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )

    if not results:
        print("No relevant results found.")
        return

    print(f"\nSearch results for: \"{args.query}\"\n")
    for i, result in enumerate(results, 1):
        print(f"{'─' * 60}")
        print(f"  #{i}  Score: {result['score']:.4f}  |  Source: {result['source']}")
        print(f"{'─' * 60}")
        # Show first 300 chars of content
        content_preview = result["content"][:300].strip()
        print(f"  {content_preview}")
        if len(result["content"]) > 300:
            print(f"  ...")
        print()


if __name__ == "__main__":
    main()
```

---

# Vector Databases — Production Infrastructure

## Why Not Just Use NumPy?

The brute-force search above (computing dot product against every vector) works for small collections but breaks down at scale:

| Collection Size | Brute Force Search Time | ANN (HNSW) Search Time |
| --------------- | ----------------------- | ---------------------- |
| 1,000           | <1ms                    | <1ms                   |
| 10,000          | ~1ms                    | <1ms                   |
| 100,000         | ~10ms                   | ~1ms                   |
| 1,000,000       | ~100ms                  | ~2ms                   |
| 10,000,000      | ~1s                     | ~5ms                   |
| 100,000,000     | ~10s                    | ~10ms                  |

Brute force is O(n) — you compare the query against every vector. Approximate Nearest Neighbor (ANN) algorithms like HNSW are O(log n) — they use smart data structures to skip most comparisons.

## Vector Database Options

| Database       | Type              | ANN Algorithm | Filtering | Cost Model       | Best For                            |
| -------------- | ----------------- | ------------- | --------- | ---------------- | ----------------------------------- |
| **pgvector**   | Postgres extension| HNSW, IVFFlat | SQL WHERE  | Self-hosted      | Teams already using Postgres        |
| **Pinecone**   | Managed SaaS      | Proprietary   | Metadata   | Per-vector/query | Serverless, minimal ops             |
| **Qdrant**     | Open source        | HNSW          | Rich filters| Self-hosted/cloud| Production, rich filtering needs   |
| **Weaviate**   | Open source        | HNSW          | GraphQL    | Self-hosted/cloud| Hybrid search built-in              |
| **Milvus**     | Open source        | HNSW, IVFFlat | Expressions| Self-hosted/cloud| Very large scale (billions)         |
| **Chroma**     | Open source        | HNSW          | Metadata   | Self-hosted      | Local dev, prototyping              |

### pgvector — The Pragmatic Choice

If you are already running Postgres, pgvector is the simplest path to production vector search:

```sql
-- Enable the extension
CREATE EXTENSION vector;

-- Create a table with a vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536)  -- 1536 dimensions
);

-- Insert a document with its embedding
INSERT INTO documents (content, source, embedding)
VALUES (
    'CrashLoopBackOff debugging guide...',
    'k8s-troubleshooting',
    '[0.023, -0.041, 0.089, ...]'  -- 1536 floats
);

-- Create an HNSW index for fast search
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Semantic search query
SELECT
    content,
    source,
    1 - (embedding <=> query_embedding) AS similarity
FROM documents
ORDER BY embedding <=> '[0.034, -0.052, 0.076, ...]'  -- query vector
LIMIT 5;
```

```python
# Python with pgvector
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)
cur = conn.cursor()


def search_pgvector(
    query_embedding: list[float],
    top_k: int = 5,
    namespace: str | None = None,
) -> list[dict]:
    """Search documents using pgvector."""

    sql = """
        SELECT content, source, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
    """
    params = [query_embedding]

    if namespace:
        sql += " WHERE source LIKE %s"
        params.append(f"{namespace}%")

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([query_embedding, top_k])

    cur.execute(sql, params)
    rows = cur.fetchall()

    return [
        {
            "content": row[0],
            "source": row[1],
            "metadata": row[2],
            "similarity": float(row[3]),
        }
        for row in rows
    ]
```

> [!TIP]
> **pgvector is the recommended starting point for most teams.** It requires no new infrastructure — just a Postgres extension. It supports HNSW indexing for fast search up to ~5M vectors. Only move to a dedicated vector database when you need >10M vectors or sub-millisecond search latency.

---

# Chunking Strategies — A Critical Design Decision

## Why Chunking Matters

Embedding models have token limits (typically 512–8192 tokens). Documents are often much longer. You must split documents into chunks before embedding.

The way you chunk directly affects search quality:

- **Chunks too small** → lose context, semantically incomplete
- **Chunks too large** → dilute meaning, poor retrieval precision
- **No overlap** → miss information at chunk boundaries
- **Wrong split points** → break sentences and concepts mid-thought

## Chunking Strategies Comparison

| Strategy                  | Chunk Size | Overlap | Quality | Complexity | Best For                            |
| ------------------------- | ---------- | ------- | ------- | ---------- | ----------------------------------- |
| Fixed character split     | 500–1000   | 100–200 | Basic   | Simple     | Quick prototyping                   |
| Sentence-based            | Varies     | 0–1     | Good    | Medium     | Well-structured prose               |
| Paragraph-based           | Varies     | 0       | Good    | Simple     | Documentation with clear paragraphs |
| Recursive (by separators) | 500–1000   | 100     | Good    | Medium     | General purpose (LangChain default) |
| Semantic chunking         | Varies     | 0       | Best    | Complex    | High-quality RAG                    |
| Markdown header-based     | Varies     | 0       | Best    | Medium     | Markdown documentation              |

### Markdown-Aware Chunking

For Kubernetes documentation (which is mostly Markdown), header-based chunking preserves the most semantic structure:

```python
import re


def chunk_markdown(
    text: str,
    source: str = "",
    max_chunk_size: int = 1000,
) -> list[DocumentChunk]:
    """
    Chunk Markdown by headers, preserving section structure.
    Each section (## or ###) becomes a chunk.
    Oversized sections are split further by paragraphs.
    """
    chunks = []
    chunk_index = 0

    # Split by headers (## and ###)
    sections = re.split(r'\n(?=#{1,3} )', text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_chunk_size:
            chunks.append(DocumentChunk(
                id=f"{source}:chunk_{chunk_index}",
                source=source,
                content=section,
                metadata={
                    "chunk_index": chunk_index,
                    "header": section.split("\n")[0] if section.startswith("#") else "",
                },
            ))
            chunk_index += 1
        else:
            # Split oversized sections by paragraphs
            paragraphs = section.split("\n\n")
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                    chunks.append(DocumentChunk(
                        id=f"{source}:chunk_{chunk_index}",
                        source=source,
                        content=current_chunk.strip(),
                        metadata={"chunk_index": chunk_index},
                    ))
                    chunk_index += 1
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk.strip():
                chunks.append(DocumentChunk(
                    id=f"{source}:chunk_{chunk_index}",
                    source=source,
                    content=current_chunk.strip(),
                    metadata={"chunk_index": chunk_index},
                ))
                chunk_index += 1

    return chunks
```

---

# Distributed Systems Perspective

## Embeddings at Scale

### Indexing Pipeline

```
Document Sources              Embedding Pipeline              Vector Store
                                                              
┌──────────┐                                                  
│ Docs DB  │──┐                                               
└──────────┘  │                                               
              │   ┌──────────┐   ┌──────────┐   ┌──────────┐
┌──────────┐  ├──▶│ Chunker  │──▶│ Embedder │──▶│ Vector   │
│ Git Repos│──┤   │          │   │ (batch)  │   │ Database │
└──────────┘  │   │ Split    │   │          │   │          │
              │   │ Clean    │   │ OpenAI   │   │ pgvector │
┌──────────┐  │   │ Dedupe   │   │ Voyage   │   │ Pinecone │
│ Wikis    │──┘   └──────────┘   └──────────┘   │ Qdrant   │
└──────────┘                                     └──────────┘
                                                      │
                                                      ▼
                                                ┌──────────┐
                                                │ Query    │
                                                │ Service  │
                                                └──────────┘
```

### Concerns at Scale

| Concern                       | Problem                                    | Solution                                      |
| ----------------------------- | ------------------------------------------ | --------------------------------------------- |
| Re-indexing cost              | Millions of docs × embedding API cost      | Incremental updates, change detection         |
| Embedding model updates       | New model = re-embed everything             | Version-tagged collections, gradual migration |
| Stale embeddings              | Documents change but embeddings do not      | Event-driven re-indexing pipeline             |
| API rate limits               | Embedding APIs have rate limits              | Batching + rate limiter + queuing             |
| Storage growth                | Vectors accumulate as documents grow         | TTL policies, garbage collection              |
| Multi-tenancy                 | Different customers need isolated indexes   | Collection per tenant, metadata filtering     |

### Incremental Indexing

```python
import hashlib


def document_fingerprint(content: str) -> str:
    """Generate a hash of document content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()


class IncrementalIndexer:
    """Only re-embed documents that have changed."""

    def __init__(self, engine: SemanticSearchEngine):
        self.engine = engine
        self.fingerprints: dict[str, str] = {}  # source → hash

    def update(self, documents: list[dict[str, str]]) -> dict:
        """Update the index, only re-embedding changed documents."""
        stats = {"added": 0, "updated": 0, "unchanged": 0}

        docs_to_embed = []

        for doc in documents:
            source = doc["source"]
            fingerprint = document_fingerprint(doc["content"])

            if source in self.fingerprints:
                if self.fingerprints[source] == fingerprint:
                    stats["unchanged"] += 1
                    continue
                else:
                    # Document changed — remove old chunks, re-embed
                    self._remove_chunks(source)
                    stats["updated"] += 1
            else:
                stats["added"] += 1

            docs_to_embed.append(doc)
            self.fingerprints[source] = fingerprint

        if docs_to_embed:
            self.engine.add_documents(docs_to_embed)

        return stats
    
    def _remove_chunks(self, source: str) -> None:
        """Remove all chunks from a given source."""
        self.engine.chunks = [
            c for c in self.engine.chunks
            if c.source != source
        ]
        # Rebuild the numpy matrix
        if self.engine.chunks:
            self.engine.embeddings = np.array(
                [c.embedding for c in self.engine.chunks]
            )
        else:
            self.engine.embeddings = None
```

---

# Infrastructure Perspective

## Deploying Embedding Search on Kubernetes

```yaml
# pgvector deployment for vector search
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-pgvector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-pgvector
  template:
    spec:
      containers:
        - name: postgres
          image: pgvector/pgvector:pg16
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: embeddings
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: pg-secrets
                  key: username
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: pg-secrets
                  key: password
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2000m
              memory: 4Gi
          volumeMounts:
            - name: pgdata
              mountPath: /var/lib/postgresql/data
      volumes:
        - name: pgdata
          persistentVolumeClaim:
            claimName: pgvector-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pgvector-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

---

# Security Perspective

## Embedding Security Concerns

| Concern                    | Risk                                            | Mitigation                                     |
| -------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| Data leakage via embeddings | Embeddings encode semantic information           | Tenant isolation, access control on vector DB  |
| Embedding inversion attacks | Reconstructing original text from embeddings     | Low practical risk, but don't expose raw vectors to untrusted clients |
| PII in embeddings          | Embeddings of PII-containing text retain meaning | Strip PII before embedding, or use dedicated PII-safe pipelines |
| Prompt injection via retrieval | Malicious content in docs gets retrieved and injected into LLM prompt | Content sanitization at index time, separate context in prompt |
| Multi-tenant data exposure | Tenant A's query retrieves Tenant B's documents  | Strict metadata filtering, separate collections per tenant |

---

# Reliability Perspective

## Common Failures

| Failure Mode                      | Impact                                    | Mitigation                                       |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| Embedding API timeout             | Search fails                              | Retry with backoff, cache recent query embeddings |
| Stale index                       | Returns outdated or deleted content       | Incremental re-indexing, TTL on chunks            |
| Vector DB outage                  | Complete search failure                   | Read replicas, fallback to keyword search         |
| Model version mismatch            | Query vector incompatible with stored vectors | Version tag collections, migration process     |
| Chunk boundary artifacts          | Important context split across chunks     | Overlap, header-aware chunking                   |
| Low recall                        | Relevant docs not returned                | Hybrid search, re-ranking, chunk size tuning     |

---

# Evaluation Perspective

## How to Evaluate Embedding Search Quality

### Retrieval Metrics

| Metric             | Formula                                              | What It Measures                                |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| **Recall@K**        | (relevant docs in top K) / (total relevant docs)     | Did we find what we should have found?           |
| **Precision@K**     | (relevant docs in top K) / K                         | Are the returned results actually relevant?      |
| **MRR**             | 1 / (rank of first relevant result)                  | How quickly do we surface the best result?       |
| **NDCG@K**          | Normalized discounted cumulative gain                | Are relevant results ranked higher?              |
| **Hit Rate**        | (queries with at least 1 relevant result in top K) / total queries | Coverage across query types |

### Building an Evaluation Set

```python
EVAL_SET = [
    {
        "query": "Why is my pod crashing?",
        "relevant_sources": [
            "k8s-troubleshooting/crashloopbackoff",
            "k8s-troubleshooting/oomkilled",
        ],
    },
    {
        "query": "How do I expose my service to the internet?",
        "relevant_sources": [
            "k8s-networking/service-types",
        ],
    },
    {
        "query": "Container image cannot be pulled",
        "relevant_sources": [
            "k8s-troubleshooting/imagepullbackoff",
        ],
    },
    {
        "query": "Pod stuck in pending state, won't schedule",
        "relevant_sources": [
            "k8s-troubleshooting/pending-pods",
        ],
    },
]


def evaluate_search(
    engine: SemanticSearchEngine,
    eval_set: list[dict],
    top_k: int = 5,
) -> dict:
    """Evaluate search quality against a labeled dataset."""
    total_recall = 0
    total_precision = 0
    total_mrr = 0
    hits = 0

    for case in eval_set:
        results = engine.search(case["query"], top_k=top_k)
        result_sources = [r["source"] for r in results]
        relevant = set(case["relevant_sources"])

        # Recall@K
        found = sum(1 for s in result_sources if s in relevant)
        recall = found / len(relevant) if relevant else 0
        total_recall += recall

        # Precision@K
        precision = found / len(result_sources) if result_sources else 0
        total_precision += precision

        # MRR
        for rank, source in enumerate(result_sources, 1):
            if source in relevant:
                total_mrr += 1 / rank
                break

        # Hit Rate
        if found > 0:
            hits += 1

    n = len(eval_set)
    return {
        "recall@k": total_recall / n,
        "precision@k": total_precision / n,
        "mrr": total_mrr / n,
        "hit_rate": hits / n,
        "eval_cases": n,
        "top_k": top_k,
    }


# Run evaluation
metrics = evaluate_search(engine, EVAL_SET, top_k=3)
print(f"Recall@3:    {metrics['recall@k']:.3f}")
print(f"Precision@3: {metrics['precision@k']:.3f}")
print(f"MRR:         {metrics['mrr']:.3f}")
print(f"Hit Rate:    {metrics['hit_rate']:.3f}")
```

---

# Observability

## What to Monitor

| Metric                         | Why It Matters                                  | Implementation                    |
| ------------------------------ | ----------------------------------------------- | --------------------------------- |
| Embedding API latency          | Affects search response time                    | Timer per API call                |
| Search latency (vector DB)     | Core user-facing performance metric             | Timer per search query            |
| Recall quality (sampled)       | Detect degradation in search quality            | Periodic evaluation against test set |
| Index size (vector count)      | Capacity planning, cost forecasting             | Counter from vector DB            |
| Cache hit rate                 | Efficiency of query embedding cache             | Counter (hit/miss)                |
| Embedding API error rate       | Detect API issues or rate limiting              | Error counter                     |
| Top similarity scores          | Detect if queries are finding relevant content  | Histogram of top-1 similarity scores |

---

# Connections

## How Embeddings Connect to Other AI Engineering Topics

| Topic                  | Connection                                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Tokenization**       | Embedding models tokenize input text before processing. Token limits determine max chunk size.                  |
| **RAG**                | Embeddings are the retrieval mechanism in RAG. They find relevant context to inject into the LLM prompt.        |
| **Vector Databases**   | Store and search embedding vectors at scale. The infrastructure layer for embedding-based systems.               |
| **Prompt Engineering** | Retrieved chunks become part of the prompt. How you format and inject them affects LLM output quality.          |
| **Tool Calling**       | Semantic search can be exposed as a tool — the model calls a "search" tool to retrieve context.                |
| **Fine-Tuning**        | You can fine-tune embedding models on your domain data to improve retrieval quality.                            |
| **Agents**             | Agents use embeddings for long-term memory (semantic retrieval of past interactions).                           |
| **Evaluation**         | Retrieval quality evaluation (recall, precision, MRR) is a prerequisite for evaluating RAG quality.            |
| **Streaming**          | Retrieval happens before generation. The retrieval step adds latency before the stream starts.                  |
| **AI Memory**          | Embedding-based memory stores past interactions and retrieves relevant ones based on semantic similarity.       |
| **MCP**                | MCP resource providers can expose semantic search as a standard tool for any AI client.                        |

---

# Expert Insights

## What Beginners Think

- "Embeddings are just vectors — they're simple."
- "I can use any embedding model for any task."
- "Bigger chunks are always better for context."
- "Cosine similarity is all I need."

## What Experienced AI Engineers Know

- Embeddings encode **meaning**, but the quality depends heavily on the model, the chunking strategy, and the domain fit. The same query can produce dramatically different results with different models.
- **Chunking is the most underrated design decision** in any embedding-based system. Bad chunking ruins search quality no matter how good your embedding model is.
- **Hybrid search** (keyword + semantic) always outperforms pure semantic search. Always implement both.
- You must use the **same model for indexing and querying**. Switching models requires complete re-indexing.
- Embedding costs are negligible. Vector storage and search infrastructure are the real costs.

## What Staff Engineers Optimize For

- **Retrieval pipeline design** — optimizing the entire path from query to relevant context, including chunking, embedding, indexing, search, and re-ranking.
- **Index lifecycle management** — incremental updates, staleness detection, model migration, multi-version indexes.
- **Cost per query** — balancing embedding dimensions, index size, and search algorithm complexity.
- **Multi-tenancy** — isolating tenant data at the vector database level while sharing infrastructure.
- **Evaluation-driven development** — never shipping a chunking or model change without running retrieval benchmarks.

## What Frontier AI Labs Care About

- Embedding models optimized for specific domains (code, legal, medical).
- Late-interaction models (ColBERT) for higher-quality retrieval without the latency of cross-encoders.
- Matryoshka embeddings — models that produce useful vectors at multiple dimension sizes from a single forward pass.
- Learned sparse embeddings that combine the benefits of keyword and semantic search in a single vector.
- Real-time embedding index updates for rapidly changing knowledge bases.

## Common Interview Trap

**Interviewer:** "How would you implement semantic search for a knowledge base?"

**Weak answer:** "I would embed all documents and use cosine similarity to find the most similar ones."

**Excellent answer:** "I would implement a complete retrieval pipeline: (1) chunk documents using a content-aware strategy — for Markdown docs, split by headers with overlap — targeting 300-500 token chunks, (2) embed chunks using a model like text-embedding-3-small for cost efficiency or voyage-3-large for quality, (3) store vectors in pgvector for simplicity or Qdrant for scale, with HNSW indexing, (4) implement hybrid search combining BM25 keyword search with vector similarity using Reciprocal Rank Fusion, (5) add metadata filtering for scoping results, (6) build an evaluation set with labeled query-document pairs and measure Recall@5 and MRR, (7) set up incremental indexing with content hashing to avoid re-embedding unchanged documents. The chunking strategy and hybrid search are the two decisions that affect quality the most — more so than the embedding model choice."

---

# Production Failure Modes

| Failure Mode                       | Impact                                    | Mitigation                                         |
| ---------------------------------- | ----------------------------------------- | -------------------------------------------------- |
| Wrong embedding model for query    | Zero or garbage results                   | Validate model consistency at index/query time      |
| Chunks too large                   | Low retrieval precision, diluted meaning  | Benchmark different chunk sizes, target 300-500 tokens |
| Chunks too small                   | Low recall, fragmented context             | Add overlap, use larger chunks for conceptual content |
| No overlap between chunks          | Information lost at boundaries            | Use 10-20% overlap                                  |
| Stale index                        | Outdated or deleted content returned       | Incremental re-indexing, content hashing             |
| Embedding API rate limiting        | Indexing pipeline stalls                   | Batching, rate limiter, exponential backoff          |
| Vector DB capacity exceeded        | Search degrades or fails                   | Monitoring, capacity planning, horizontal scaling    |
| Low similarity threshold           | Too many irrelevant results               | Tune threshold per use case (0.5-0.7 typical)        |
| High similarity threshold          | Misses relevant but differently worded docs | Lower threshold, add hybrid search                  |
| Model version change               | Vectors become incompatible                | Version-tagged collections, migration pipeline       |

---

# Evolution of the Field

### Historical Approach

```
2013: Word2Vec — first practical word embeddings
  ↓
2014: GloVe — global vectors for word representation
  ↓
2018: ELMo, BERT — contextual embeddings (word meaning depends on context)
  ↓
2019: Sentence-BERT — embeddings for sentences, not just words
  ↓
2022: text-embedding-ada-002 (OpenAI) — simple API, good quality
  ↓
2024: text-embedding-3-* (OpenAI) — dimension reduction, better quality
      Voyage AI, Cohere Embed v3 — specialized embedding models
  ↓
2025: Matryoshka embeddings, late-interaction models (ColBERT),
      hybrid sparse-dense embeddings, long-context embeddings (128K+ tokens)
  ↓
Emerging: Multi-modal embeddings (text + image + code in same space),
          real-time index updates, learned sparse representations
```

### Established Practice

- API-based embeddings (OpenAI, Voyage, Cohere) for most use cases
- pgvector or dedicated vector DBs for storage and search
- HNSW algorithm for approximate nearest neighbor search
- Hybrid search (keyword + semantic) for production retrieval
- Chunking with overlap for document processing

### Emerging Practice

- Matryoshka embeddings for flexible dimension sizing
- ColBERT / late-interaction models for higher-quality retrieval
- Binary quantization for 32x storage reduction with minimal quality loss
- Multimodal embeddings for unified text + image search

### Research Direction

- Can we build embedding models that understand domain-specific jargon without fine-tuning?
- How do we handle embedding drift as models are updated?
- What is the theoretical limit of retrieval quality for a given embedding dimension?
- Can we build lossless compression for embedding vectors?

---

# Hands-On Project — Kubernetes Documentation Semantic Search

## Problem Statement

Build a semantic search engine that indexes Kubernetes documentation and answers troubleshooting queries by finding the most relevant documentation chunks.

## Folder Structure

```
k8s-semantic-search/
├── pyproject.toml
├── src/
│   └── k8s_search/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point
│       ├── chunker.py          # Document chunking strategies
│       ├── embedder.py         # Embedding API wrapper
│       ├── search_engine.py    # SemanticSearchEngine class
│       ├── vector_store.py     # Abstract vector store (numpy / pgvector)
│       └── evaluation.py       # Search quality evaluation
├── docs/
│   ├── crashloopbackoff.md
│   ├── oomkilled.md
│   ├── pending-pods.md
│   ├── service-types.md
│   └── imagepullbackoff.md
├── tests/
│   ├── test_chunker.py
│   ├── test_search.py
│   └── test_evaluation.py
├── eval/
│   └── test_queries.json       # Labeled evaluation set
└── README.md
```

## Architecture Diagram

```
Markdown Docs
    │
    ▼
┌──────────────────────┐
│  Markdown Chunker     │  Split by headers, overlap
├──────────────────────┤
│  Embedding API        │  text-embedding-3-small
├──────────────────────┤
│  Vector Store         │  NumPy (dev) / pgvector (prod)
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │  Index File  │  k8s_search_index.json
    └──────┬──────┘
           │
    User Query
    │
    ▼
┌──────────────────────┐
│  Query Embedder       │  Same model as indexing
├──────────────────────┤
│  Vector Search        │  Cosine similarity, top-K
├──────────────────────┤
│  Result Formatter     │  Score, source, content preview
└──────────────────────┘
    │
    ▼
  Ranked Results
```

## Implementation Steps

1. Implement `chunk_markdown()` with header-aware splitting and overlap
2. Implement `SemanticSearchEngine` with embed, index, search, save, load
3. Index sample Kubernetes troubleshooting documentation
4. Build the CLI with search and index commands
5. Build an evaluation set with 10+ labeled query-document pairs
6. Measure Recall@3, Precision@3, and MRR
7. Experiment with chunk sizes and compare metrics

## Production Improvements

- Replace numpy brute-force with pgvector HNSW index
- Add hybrid search with BM25 keyword matching
- Implement incremental indexing with content hashing
- Add re-ranking with a cross-encoder model
- Implement query embedding caching
- Add metadata filtering (namespace, resource type, severity)

## Monitoring

- Embedding API latency and error rate
- Search latency (p50, p95, p99)
- Top similarity scores distribution
- Index size and growth rate
- Recall@K sampled periodically against evaluation set
- Alert on recall dropping below 0.8

## Testing Strategy

- **Unit tests**: Chunking logic, cosine similarity, evaluation metrics
- **Integration tests**: Embedding API calls, end-to-end search
- **Evaluation tests**: Recall@K and MRR against labeled test set
- **Regression tests**: When changing chunk size, model, or strategy

---

# Exercises

## Beginner Exercises

1. Embed two sentences and compute their cosine similarity manually (with NumPy)
2. Embed 5 Kubernetes-related sentences and 5 unrelated sentences — verify that related sentences have higher similarity scores
3. Build a minimal search function that embeds a query and finds the most similar document from a list of 10
4. Experiment with different queries for the same concept ("pod crashing" vs "container restart loop" vs "CrashLoopBackOff") — do they find the same results?
5. Save and load an embedding index to/from a JSON file

## Intermediate Exercises

1. Implement chunk overlap and compare search quality with and without overlap
2. Build a Markdown-aware chunker that splits on headers instead of character count
3. Implement batch embedding to index 100 documents efficiently
4. Compare search results from `text-embedding-3-small` vs `text-embedding-3-large` — is the quality difference worth the cost?
5. Implement dimension reduction (256 vs 1536 vs 3072) and measure quality impact

## Advanced Exercises

1. Implement hybrid search combining BM25 keyword search with vector similarity using Reciprocal Rank Fusion
2. Build an evaluation framework with Recall@K, Precision@K, and MRR — run it on a labeled test set
3. Implement incremental indexing with content hashing to avoid re-embedding unchanged documents
4. Set up pgvector in a Docker container and migrate from NumPy brute-force to HNSW-indexed search
5. Implement a re-ranking step using a cross-encoder model to improve precision of the top results

## Production Engineering Exercises

1. Design an embedding pipeline that processes 1 million documents with rate limiting, error handling, and progress tracking
2. Implement multi-tenant vector search — each tenant has isolated data but shares the same pgvector database
3. Build a monitoring dashboard that tracks embedding latency, search latency, index size, and recall over time
4. Design a migration strategy for switching from text-embedding-3-small to a new embedding model without downtime
5. Implement query embedding caching with a TTL — cache recent query embeddings to avoid redundant API calls

## System Design Exercises

1. **Design the retrieval layer for a production RAG system** serving 10,000 queries per day over 500,000 documents. Consider: chunking strategy, embedding model selection, vector database choice, hybrid search, caching, and scaling.
2. **Design a semantic code search engine** (like GitHub code search or Cursor's codebase indexing). How do you chunk code? What embedding model do you use? How do you handle multiple programming languages? How do you keep the index fresh as code changes?
3. **Design a multi-modal search system** that searches across text documents, images, and code snippets. How do you handle different modalities in the same vector space?

---

# Key Takeaways

1. **An embedding is coordinates representing meaning** — a fixed-length vector where semantically similar texts have similar vectors. This is the foundation of all modern AI search systems.
2. **Cosine similarity measures semantic distance** — it computes the angle between two vectors. Small angle = similar meaning. Most embedding APIs return normalized vectors, making cosine similarity equal to the dot product.
3. **Semantic search finds meaning, not keywords** — "pod keeps dying" matches "CrashLoopBackOff troubleshooting" even though they share no words. This is the killer feature of embeddings.
4. **Hybrid search always wins** — combining keyword search (BM25) with semantic search (embeddings) outperforms either alone. This is an established best practice for production systems.
5. **Chunking is the most underrated design decision** — bad chunks ruin search quality regardless of embedding model quality. Use content-aware chunking (header-based for Markdown, paragraph-based for prose) with 10-20% overlap.
6. **Same model for indexing and querying** — vectors from different models are incompatible. Switching models requires complete re-indexing.
7. **Embedding costs are negligible** — $0.10 to embed 10,000 documents. The expensive part of RAG is the LLM generation step, not embedding. Optimize LLM costs first.
8. **pgvector is the pragmatic starting point** — if you already run Postgres, add the pgvector extension. Only move to dedicated vector databases when you exceed ~5M vectors or need sub-millisecond latency.
9. **Evaluation is non-negotiable** — measure Recall@K, Precision@K, and MRR against a labeled test set. Never ship a chunking or model change without running retrieval benchmarks.
10. **Embeddings are the retrieval layer for RAG** — they find the right information to put in the LLM's context window. RAG quality is bounded by retrieval quality. If you cannot find the right documents, the LLM cannot generate good answers.
11. **Dimension reduction is a cost optimization lever** — OpenAI's text-embedding-3 models support reducing dimensions from 3072 to 256 with minimal quality loss. Less storage, faster search, lower cost.
12. **Incremental indexing prevents waste** — hash document content to detect changes. Only re-embed documents that have actually changed. This is critical for large knowledge bases.
13. **Vector databases use ANN algorithms (HNSW)** — approximate nearest neighbor search is O(log n) vs. brute force O(n). At 1M+ vectors, the difference is 100x in latency.
14. **Security matters** — embeddings encode semantic information. Implement tenant isolation at the vector database level. Never expose raw vectors to untrusted clients.
15. **Embeddings are the foundation of AI memory, RAG, and semantic tools** — mastering embeddings unlocks the next three layers of the AI engineering stack.
