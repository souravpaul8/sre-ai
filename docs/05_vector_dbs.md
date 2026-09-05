# Day 5 — Vector Databases

## Goal

Understand how meaning is stored and retrieved at scale.

Not the marketing.

Not the "just use Pinecone."

The engineering.

A vector database stores **high-dimensional embedding vectors** and answers *"which items are semantically closest to this query?"* in milliseconds, at scale, with metadata filtering. It is the **storage and retrieval backbone** of every RAG pipeline, every semantic search product, every AI agent with memory.

Think of it as the bridge between **embeddings** (Day 4 — turning meaning into coordinates) and **retrieval** (Day 6 — finding the right context for the LLM). Without a vector database, your embeddings are just floating-point numbers sitting in a JSON file.

This is where distributed systems, data structures, and AI engineering meet. Understanding vector databases deeply means understanding **ANN algorithms, index tuning, filtering strategies, memory vs. recall tradeoffs, and multi-tenant isolation** — the kind of systems work that separates AI prototypes from AI products.

---

## Original Notes (Preserved)

**Learn — Choose one**

- pgvector
- Qdrant
- Weaviate
- Milvus

**Recommendation:** **Qdrant** or **pgvector**

**Concepts**

- Collections
- Indexes
- Search
- Metadata
- Filtering

**Deliverable:** Kubernetes Documentation Search — upload Kubernetes Docs, Helm Docs, Prometheus Docs. Search naturally.

---

## Big Picture

### Where Vector Databases Fit in the AI Stack

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
│         Retrieval Layer                 │
│   (Embeddings, Vector Search,           │  ← You are here (Day 5)
│    Semantic Matching, Filtering)        │
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

Vector databases sit at the **retrieval layer** — they are the infrastructure that makes embedding-based search possible at production scale. Day 4 taught you how to turn text into vectors. Today you learn how to **store, index, and search** millions of those vectors efficiently.

### Why This Concept Exists

LLMs have three hard limits:

1. **Finite context window** — you cannot paste all of the Kubernetes docs into a prompt.
2. **Frozen knowledge** — the model does not know your private/internal/latest docs.
3. **Hallucination** — without grounding, the model invents plausible-but-wrong answers.

A vector database solves the *retrieval* half of "retrieve-then-generate." It lets you take a natural-language question, convert it to a vector, and find the **k most semantically similar** document chunks — even when there is **zero keyword overlap** (e.g., query "pod keeps restarting" retrieving a doc titled "CrashLoopBackOff").

### Why Frontier AI Companies Care

Every knowledge-intensive AI product depends on vector search:

- **Perplexity** — vector search over web page embeddings to find sources for answers
- **Cursor** — embeds your entire codebase, vector search for semantic code retrieval
- **ChatGPT** — vector search powers file search in GPTs and Projects
- **GitHub Copilot** — vector search over repository context for code suggestions
- **Notion AI** — vector search over workspace content for Q&A
- **Glean** — enterprise search across all company knowledge via vector retrieval

Without vector databases, these products cannot work. Retrieval quality is usually the **#1 driver of end-to-end answer quality** — a better retriever beats a bigger model.

Vector search at scale (billions of vectors, <50ms p99) is a genuine distributed-systems problem, which is why it shows up in senior interviews.

If you do not understand vector databases deeply, you cannot:

- Build production RAG systems
- Debug retrieval quality issues
- Optimize search latency and memory usage
- Design multi-tenant vector search
- Make informed infrastructure choices (pgvector vs. Qdrant vs. Milvus)
- Pass an AI engineering or systems interview

---

## Mental Model

### The Software Engineer's Model of Vector Databases

Think of a vector database as a **`GROUP BY nearest-neighbor` index for meaning**.

Where a Postgres B-tree index answers "rows WHERE id = 5", a vector index answers "rows ORDER BY semantic_distance(embedding, query) LIMIT 10". It is a **spatial index (like a geo/R-tree) but in 768–3072 dimensions**.

```
Traditional SQL:
  SELECT * FROM docs WHERE id = 42;
  → B-tree index → O(log N) → exact match

Vector Database:
  SELECT * FROM docs ORDER BY distance(embedding, query_vec) LIMIT 10;
  → HNSW/IVF index → O(log N) → approximate nearest neighbors
```

### The Library Analogy

Imagine a massive library with millions of books:

```
BRUTE FORCE (no index):
  "Find books about Kubernetes pod crashes"
  → Walk past EVERY shelf, read EVERY book title, compare meaning
  → Hours to complete
  → Always finds the best matches (100% recall)

TRADITIONAL INDEX (keyword):
  "Find books about Kubernetes pod crashes"
  → Look up "Kubernetes", "pod", "crashes" in card catalog
  → Fast, but misses a book titled "CrashLoopBackOff Debugging Guide"
  → Keyword mismatch

VECTOR INDEX (ANN):
  "Find books about Kubernetes pod crashes"
  → Go to the "meaning neighborhood" of that query
  → Check nearby shelves (not ALL shelves)
  → Finds "CrashLoopBackOff Debugging Guide" because it's semantically close
  → Fast AND semantic
  → Might miss 1-2 matches in distant corners (approximate, ~95-99% recall)
```

### The Distance = Difference in Meaning

```
query_vec = embed("pod keeps crashing")

doc_a = embed("CrashLoopBackOff troubleshooting")  → distance: 0.12  ✅ Very close
doc_b = embed("Container restart debugging")        → distance: 0.18  ✅ Close
doc_c = embed("Kubernetes service types")            → distance: 0.65  ✗  Different topic
doc_d = embed("How to bake a cake")                  → distance: 0.94  ✗  Unrelated
```

Small distance = similar meaning. Large distance = different meaning.

The vector database finds these nearest neighbors **without comparing against every single vector** — that's the entire point of the ANN index.

---

## Core Concepts — Deep Dive

### 1. Collections

A **collection** (a.k.a. index / class / table) is a named group of vectors that all share the **same dimensionality and distance metric**. Analogous to a SQL table.

```
Collection: "k8s_docs"
├── Dimensionality: 1536
├── Distance Metric: Cosine
├── Vector Count: 50,000
├── Index Type: HNSW
└── Payload Schema:
    ├── source: string
    ├── url: string
    ├── heading: string
    └── version: string
```

You create separate collections for:
- Different embedding models (they produce incompatible vector spaces)
- Different data domains (if you want strict isolation)
- Different tenants (in a multi-tenant setup, one option)

> [!WARNING]
> You **cannot mix embedding models** in one collection. `text-embedding-3-small` (1536-dim) and `text-embedding-3-large` (3072-dim) live in different vector spaces — their distances are meaningless across models. One model = one collection (or a re-embed migration).

### 2. Indexes — The Heart of It

A brute-force scan comparing the query to *every* vector is **O(N·D)** — fine for 10k vectors, fatal for 10M. Vector indexes trade a little accuracy for enormous speed. This is **ANN (Approximate Nearest Neighbor)**.

#### Why ANN Instead of Exact?

| Collection Size | Brute Force Search Time | ANN (HNSW) Search Time | Speedup |
| --------------- | ----------------------- | ---------------------- | ------- |
| 1,000           | <1ms                    | <1ms                   | ~1x     |
| 10,000          | ~1ms                    | <1ms                   | ~1x     |
| 100,000         | ~10ms                   | ~1ms                   | 10x     |
| 1,000,000       | ~100ms                  | ~2ms                   | 50x     |
| 10,000,000      | ~1s                     | ~5ms                   | 200x    |
| 100,000,000     | ~10s                    | ~10ms                  | 1000x   |

At 1M+ vectors, the difference is 50-200x in latency. At 100M, it's 1000x. ANN makes production-scale semantic search possible.

#### Index Types Compared

| Index                                         | How It Works                                                               | Strength                                          | Weakness                                   |
| --------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------ |
| **Flat (brute force)**                        | Compare query to ALL vectors                                               | 100% recall, exact                                | O(N), slow at scale                        |
| **IVF** (Inverted File)                       | Cluster vectors into *nlist* buckets, search only *nprobe* nearest buckets | Fast, tunable                                     | Needs training, recall drops at boundaries |
| **HNSW** (Hierarchical Navigable Small World) | Multi-layer graph you "greedy-walk" toward the query                       | Best latency/recall tradeoff, incremental inserts | High RAM, build cost                       |
| **PQ / IVF-PQ** (Product Quantization)        | Compress vectors into quantized codes                                      | Huge memory savings                               | Lower recall, lossy                        |
| **DiskANN**                                   | Graph index that lives on SSD                                              | Billion-scale on one box                          | More complex                               |

#### How HNSW Works — Visual Walkthrough

HNSW is the modern default. Understanding it is essential.

```
Layer 2 (sparse — few nodes, long-range connections):

    A ─────────────────────── B
    │                         │
    │     (skip to region)    │
    │                         │

Layer 1 (medium density):

    A ── C ── D ──── B
    │    │    │      │
    │    E    F      │

Layer 0 (dense — all nodes, short-range connections):

    A─C─G─E─D─H─F─I─B─J─K─L─M─N─O─P
              ▲
        query lands here
        (greedy search found E as nearest)
```

**How search works:**

1. Start at the **top layer** (sparse). Greedily walk to the node closest to the query.
2. Drop down to the **next layer** (more nodes). Continue greedy walking.
3. Keep dropping until **layer 0** (all nodes). The final greedy walk gives the nearest neighbors.

**Why it's fast:** The top layers act as "express lanes" — you quickly jump to the right neighborhood, then do fine-grained search at the bottom. You never scan all vectors.

**Parameters that matter:**

| Parameter          | When       | What it controls                          | Tradeoff                                    |
| ------------------ | ---------- | ----------------------------------------- | ------------------------------------------- |
| `M`                | Build time | Connections per node (graph connectivity) | Higher M = better recall, more RAM          |
| `ef_construction`  | Build time | Candidate list during index build         | Higher = better index quality, slower build |
| `ef_search` / `ef` | Query time | Candidate list during search              | Higher = better recall, higher latency      |

```
Low ef_search (e.g., 50):   Fast but may miss some neighbors  → 93% recall
Med ef_search (e.g., 128):  Good balance                      → 97% recall
High ef_search (e.g., 512): Slower but very thorough          → 99.5% recall
```

> [!NOTE]
> **HNSW is the modern default** (Qdrant, Weaviate, Milvus, pgvector all support it). Start with `M=16`, `ef_construction=128`, and tune `ef_search` at query time to trade recall ↔ latency.

#### How IVF Works — Visual Walkthrough

```
Training Phase (offline):
  All vectors → K-means clustering → nlist centroids (e.g., 256 clusters)

  Cluster 1: ●●●●●●●
  Cluster 2: ●●●●
  Cluster 3: ●●●●●●●●●●
  ...
  Cluster 256: ●●●●●

Search Phase:
  Query vector → Find nprobe nearest centroids → Search only those clusters

  Query: ★
    → Nearest centroids: Cluster 3, Cluster 7, Cluster 42  (nprobe=3)
    → Search ~3% of all vectors instead of 100%
    → Much faster, but might miss neighbors at cluster boundaries
```

### 3. Search & Distance Metrics

The "distance" between vectors defines similarity. **You must use the metric your embedding model expects.**

| Metric             | Formula Intuition                         | Use When                                      |
| ------------------ | ----------------------------------------- | --------------------------------------------- |
| **Cosine**         | Angle between vectors (ignores magnitude) | Most text embeddings (OpenAI recommends this) |
| **Dot Product**    | Cosine × magnitudes                       | Vectors already normalized; some models       |
| **Euclidean (L2)** | Straight-line distance                    | Image / some ML embeddings                    |

```
Cosine Similarity:

    cos(θ) = (A · B) / (‖A‖ × ‖B‖)

    cos(0°)   = 1.0  → identical direction  → identical meaning
    cos(90°)  = 0.0  → perpendicular        → unrelated
    cos(180°) = -1.0 → opposite direction   → opposite meaning

For normalized vectors (OpenAI text-embedding-3-*):
    cosine_similarity = dot_product  (massive optimization — no magnitude calc needed)
```

> [!WARNING]
> Using the **wrong metric** for your model silently tanks recall. OpenAI `text-embedding-3-*` models are normalized → cosine and dot product are equivalent. Always check the model card and pick what it says.

### 4. Metadata (Payload)

Alongside each vector you store a JSON **payload**:

```json
{
    "text": "CrashLoopBackOff means the container is repeatedly crashing...",
    "source": "kubernetes",
    "url": "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
    "heading": "Container restart policy",
    "version": "1.30",
    "doc_type": "troubleshooting",
    "last_updated": "2025-03-15"
}
```

This is what you return to the user (the vector itself is unreadable) and what you filter on. Without metadata, a vector database is useless — you'd retrieve vectors but have no idea what text they represent.

### 5. Filtering

Restrict search to vectors matching metadata predicates: *"nearest chunks WHERE source = 'kubernetes' AND version >= '1.29'"*.

```
Without filtering:
  Query: "CrashLoopBackOff debugging"
  → Returns chunks from Kubernetes, Helm, Prometheus, random docs
  → User only wants Kubernetes docs → irrelevant results

With filtering:
  Query: "CrashLoopBackOff debugging"  WHERE source = "kubernetes"
  → Returns only Kubernetes doc chunks
  → Precise, relevant results
```

Two strategies:

| Strategy           | How                                         | Tradeoff                                             |
| ------------------ | ------------------------------------------- | ---------------------------------------------------- |
| **Pre-filtering**  | Filter first, then ANN search the survivors | Accurate, but can be slow / break graph connectivity |
| **Post-filtering** | ANN search first, then drop non-matches     | Fast, but may return < k results                     |

> [!TIP]
> Qdrant's **filterable HNSW** integrates payload filters *into* the graph traversal, avoiding the classic "filter breaks the index" problem — a major reason it's recommended here. pgvector handles filtering via normal SQL `WHERE` clauses, which is intuitive but can be less efficient for complex filters on large datasets.

---

## Internal Architecture

### Write Path vs. Read Path

```
                    WRITE PATH                                  READ PATH
  Document                                        Natural-language query
     │                                                    │
     ▼                                                    ▼
  Chunker (split into passages)                    Embedding Model (same model!)
     │                                                    │
     ▼                                                    ▼
  Embedding Model → vector[1536]                   query_vector[1536]
     │                                                    │
     ▼                                                    ▼
  Upsert(id, vector, payload)  ──►  ┌─────────────────────────────────┐
                                     │        VECTOR DATABASE          │
                                     │  ┌───────────┐   ┌───────────┐ │
                                     │  │ HNSW/IVF  │   │  Payload  │ │
                                     │  │  Index    │   │  Store +  │ │
                                     │  │ (vectors) │   │  Filter   │ │
                                     │  └───────────┘   └───────────┘ │
                                     └─────────────────────────────────┘
                                                    │
                                          ANN search + filter
                                                    ▼
                                     top-k {id, score, payload}
```

### Inside a Vector Database Server

```
┌─────────────────────────────────────────────────────────────────┐
│                     VECTOR DATABASE SERVER                       │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   API Layer   │  │   Collection │  │   Storage Engine      │   │
│  │              │  │   Manager    │  │                      │   │
│  │  REST/gRPC   │  │              │  │  ┌────────────────┐  │   │
│  │  JSON-RPC    │  │  Create/Drop │  │  │  Vector Store   │  │   │
│  │              │  │  Schema mgmt │  │  │  (mmap files)   │  │   │
│  └──────┬───────┘  └──────┬───────┘  │  ├────────────────┤  │   │
│         │                 │          │  │  Payload Store   │  │   │
│         ▼                 ▼          │  │  (JSON/BSON)    │  │   │
│  ┌──────────────────────────────┐    │  ├────────────────┤  │   │
│  │       Query Executor          │    │  │  WAL (Write    │  │   │
│  │                              │    │  │   Ahead Log)   │  │   │
│  │  1. Parse query + filters    │    │  └────────────────┘  │   │
│  │  2. Embed (if model built-in)│    │                      │   │
│  │  3. ANN search (HNSW walk)   │    └──────────────────────┘   │
│  │  4. Apply filters            │                               │
│  │  5. Score + rank             │    ┌──────────────────────┐   │
│  │  6. Return top-k + payloads  │    │   Index Manager       │   │
│  └──────────────────────────────┘    │                      │   │
│                                      │  Build/optimize HNSW │   │
│                                      │  Handle inserts/dels │   │
│                                      │  Quantization        │   │
│                                      └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Vector Database Comparison

### The Four Choices

| Feature | pgvector | Qdrant | Weaviate | Milvus |
| --- | --- | --- | --- | --- |
| **Type** | Postgres extension | Dedicated vector DB | Dedicated vector DB | Dedicated vector DB |
| **Language** | C (Postgres) | Rust | Go | Go + C++ |
| **Index Types** | HNSW, IVFFlat | HNSW | HNSW | HNSW, IVFFlat, DiskANN, GPU |
| **Filtering** | SQL WHERE (native) | Filterable HNSW (integrated) | GraphQL filters | Expression-based |
| **Quantization** | None (community WIP) | Scalar, Product, Binary | PQ, BQ | Scalar, PQ |
| **Max Scale** | ~5M vectors (single node) | 100M+ (distributed) | 100M+ (distributed) | 1B+ (distributed, GPU) |
| **Hybrid Search** | Via pg_trgm + vector | Sparse vectors + dense | Built-in BM25 + vector | Sparse + dense |
| **Transactions** | Full ACID (it's Postgres) | Eventually consistent | Eventually consistent | Eventually consistent |
| **Ops Complexity** | Low (it's Postgres) | Medium | Medium-High | High |
| **API** | SQL | REST/gRPC | GraphQL/REST | REST/gRPC/SDK |
| **Cloud Managed** | Neon, Supabase, RDS | Qdrant Cloud | Weaviate Cloud | Zilliz Cloud |
| **Best For** | Already have Postgres; <5M vectors; want SQL | Production retrieval; rich filtering; K8s-native | Built-in modules; hybrid search out of box | Billion-scale; GPU indexing; big data teams |

### Decision Tree

```
What is your situation?
    │
    ├── Already running Postgres? → pgvector
    │   └── Need >5M vectors? → Consider Qdrant or migration
    │
    ├── Dedicated vector workload? → Qdrant
    │   └── Need K8s-native + rich filtering + quantization
    │
    ├── Want batteries-included (auto-embedding, hybrid)? → Weaviate
    │   └── Heavier to operate, more features out of box
    │
    └── Billion-scale + GPU? → Milvus
        └── Most complex, most scalable
```

> [!TIP]
> **pgvector** if you value operational simplicity and already have Postgres. **Qdrant** if retrieval is a first-class concern and you need rich filtering + quantization on Kubernetes. Both are excellent defaults; avoid Milvus unless you truly need billion-scale.

### Other Options Worth Knowing

| Database | Type | Best For |
| --- | --- | --- |
| **Pinecone** | Managed SaaS | Serverless, zero ops, per-vector pricing |
| **Chroma** | Open source (Python) | Local dev, prototyping, small scale |
| **FAISS** | Library (not a DB) | Research, embedded in your process, no persistence layer |
| **LanceDB** | Embedded (Rust) | Serverless, columnar + vector |

---

## Performance Perspective

### What Determines Search Latency

| Factor | Impact | How to Optimize |
| --- | --- | --- |
| `ef_search` (HNSW) | Higher = more candidates = higher recall + higher latency | Start at 128, tune to target recall |
| Vector dimension | Higher dim = more computation per comparison | Use dimension reduction if quality allows |
| Collection size | More vectors = deeper graph traversal | Shard across nodes |
| Payload filtering | Complex filters slow pre/post-filtering | Use filterable HNSW (Qdrant), indexed payload fields |
| Quantization | Reduces vector size = faster comparison | Use scalar quantization for 4x speedup |
| Network | Remote DB adds round-trip | Co-locate with application |

### Memory Estimation

```
Memory ≈ N × D × bytes_per_float × overhead_factor

Where:
  N = number of vectors
  D = dimensions
  bytes_per_float = 4 (float32), 2 (float16), 1 (int8/scalar quant), 0.125 (binary)
  overhead_factor = 1.5–2.0x (HNSW graph structure)

Examples (float32, 1.7x overhead):
  100K  × 1536-dim ≈  1 GB
  1M    × 1536-dim ≈  10 GB
  10M   × 1536-dim ≈  100 GB
  100M  × 1536-dim ≈  1 TB   (needs sharding or quantization)

With scalar quantization (int8, 1.7x overhead):
  1M    × 1536-dim ≈  2.5 GB   (4x reduction!)
  10M   × 1536-dim ≈  25 GB
```

### Quantization — Trading Recall for RAM

| Method | Compression | Recall Impact | Use When |
| --- | --- | --- | --- |
| **None (float32)** | 1x (baseline) | 100% baseline | <1M vectors, quality-critical |
| **Scalar (int8)** | 4x | ~1% loss | Default for production |
| **Product (PQ)** | 8–32x | 2–5% loss | Large scale, memory-constrained |
| **Binary** | 32x | 5–10% loss, needs re-scoring | Very large scale, coarse filtering |

> [!TIP]
> **Scalar quantization is the default production choice.** It gives 4x memory reduction with ~1% recall loss. Enable it from the start — don't wait until you run out of RAM.

### Benchmark Mindset

> [!IMPORTANT]
> Measure **recall@k AND p95 latency together** — either alone is meaningless. A system with 99.9% recall but 500ms p95 is useless for real-time search. A system with 1ms latency but 80% recall misses 1 in 5 relevant results.

```
Good benchmark methodology:
  1. Use production-sized data (not 10K vectors when you'll have 1M)
  2. Warm the cache (first queries are always slow)
  3. Separate embedding time from search time
  4. Report p50, p95, p99 latency (not averages)
  5. Report recall@k against brute-force ground truth
  6. Test with realistic filters (not just unfiltered search)
```

---

## Cost Perspective

### What Costs Money

| Cost Component | Driver | Optimization |
| --- | --- | --- |
| **Embedding (ingest)** | Per-token, one-time per chunk | Batch, incremental re-index |
| **Embedding (query)** | Per-token, per query | Cache query embeddings |
| **RAM (vector DB)** | N × D × bytes | Quantization (4–32x savings) |
| **Compute (search)** | CPU per query | Tune ef_search, batch queries |
| **Storage (disk)** | WAL + snapshots | Compaction, retention policies |
| **Managed pricing** | Per-vector or per-pod | Self-host at scale |

### Managed vs. Self-Hosted

| Aspect | Managed (Pinecone, Qdrant Cloud) | Self-Hosted (Qdrant/pgvector on your K8s) |
| --- | --- | --- |
| **Ops burden** | Zero | You own backups, upgrades, monitoring |
| **Cost at small scale** | Cheaper (free tiers) | More expensive (always-on infra) |
| **Cost at large scale** | Much more expensive | Cheaper |
| **Control** | Limited | Full |
| **Latency** | Extra network hop | Co-located with your app |
| **Data residency** | Vendor's cloud | Your cloud / on-prem |

### Re-Embedding Cost

Re-embedding a large corpus on every model change is expensive — version your embeddings:

```
v1: text-embedding-3-small → collection "k8s_docs_v1"
v2: text-embedding-3-large → collection "k8s_docs_v2"

Migration strategy:
  1. Build v2 in parallel (don't delete v1)
  2. Shadow-test: run queries against both, compare recall
  3. Switch traffic to v2
  4. Delete v1 after validation period
```

---

## Distributed Systems Perspective

### Sharding

```
Collection: "enterprise_docs" (100M vectors)

  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Shard 0 │   │  Shard 1 │   │  Shard 2 │   │  Shard 3 │
  │  25M vecs│   │  25M vecs│   │  25M vecs│   │  25M vecs│
  │  Node A  │   │  Node B  │   │  Node C  │   │  Node D  │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
       │              │              │              │
       └──────────────┼──────────────┼──────────────┘
                      │
                 ┌─────────┐
                 │ Query   │  Fan-out query to all shards
                 │ Router  │  Merge top-k from each
                 │         │  Return global top-k
                 └─────────┘
```

- **Scatter-gather pattern**: query fans out to all shards, each returns local top-k, router merges and returns global top-k.
- Sharding strategy: random (balanced load) or by metadata (tenant-based for isolation).

### Replication

- Copies for **high availability** and **read throughput**.
- Typically 2–3 replicas per shard.
- Read from any replica; write to primary, replicate async.

### Consistency

- Upserts are typically **eventually consistent** — a just-inserted doc may not be searchable for a moment (milliseconds to seconds).
- For most RAG use cases, this is fine (ingest is batch/async anyway).
- pgvector is the exception: full ACID transactions (it's Postgres).

### Rebuild / Compaction

- HNSW handles inserts well but **deletes leave tombstones** → periodic optimization needed.
- Qdrant: automatic segment optimization.
- pgvector: standard Postgres VACUUM.

---

## Infrastructure Perspective

### Kubernetes Deployment — Qdrant StatefulSet

```yaml
# qdrant-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: ai-search
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:v1.12.0
          ports:
            - containerPort: 6333  # REST API
              name: rest
            - containerPort: 6334  # gRPC
              name: grpc
          env:
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
          resources:
            requests:
              cpu: "1"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 16Gi
          volumeMounts:
            - name: qdrant-data
              mountPath: /qdrant/storage
          readinessProbe:
            httpGet:
              path: /readyz
              port: 6333
            initialDelaySeconds: 5
          livenessProbe:
            httpGet:
              path: /healthz
              port: 6333
            initialDelaySeconds: 10
  volumeClaimTemplates:
    - metadata:
        name: qdrant-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: ai-search
spec:
  type: ClusterIP
  selector:
    app: qdrant
  ports:
    - name: rest
      port: 6333
      targetPort: 6333
    - name: grpc
      port: 6334
      targetPort: 6334
```

### Kubernetes Deployment — pgvector

```yaml
# pgvector-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-pgvector
  namespace: ai-search
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-pgvector
  template:
    metadata:
      labels:
        app: postgres-pgvector
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
              memory: 2Gi
            limits:
              cpu: "4"
              memory: 8Gi
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
  namespace: ai-search
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 50Gi
```

### Docker Compose — Local Development

```yaml
# docker-compose.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports:
      - "6333:6333"   # REST
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: embeddings
      POSTGRES_USER: ai
      POSTGRES_PASSWORD: localdev
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes:
  qdrant_data:
  pg_data:
```

---

## Security Perspective

### Tenant Isolation

Never let tenant A retrieve tenant B's chunks. Two approaches:

| Approach | How | Tradeoff |
| --- | --- | --- |
| **Metadata filtering** | Add `tenant_id` to every payload; mandatory server-side filter on every query | Simple, shared infra; filter must be enforced server-side (never client-trusted) |
| **Collection per tenant** | Separate collection for each tenant | Strongest isolation; more ops overhead; doesn't scale to 10K+ tenants |

```python
# WRONG — client-side filter (can be bypassed)
results = client.search(query_vec, filter={"tenant_id": user.tenant_id})

# RIGHT — server-side enforcement in your API layer
def search(query: str, user: User):
    query_vec = embed(query)
    # Mandatory filter — user cannot override
    mandatory_filter = Filter(must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=user.tenant_id))
    ])
    return qdrant.search(
        collection_name="docs",
        query_vector=query_vec,
        query_filter=mandatory_filter,
        limit=10,
    )
```

### Other Security Concerns

| Concern | Risk | Mitigation |
| --- | --- | --- |
| **PII in payloads** | Payloads stored in plaintext | Redact/encrypt sensitive fields before storing |
| **Embedding inversion** | Embeddings can partially leak source text | Treat embeddings as sensitive data; don't expose raw vectors to untrusted clients |
| **Network exposure** | Vector DB API accessible without auth | TLS + auth (API keys, mTLS); don't expose to public internet |
| **Backup exposure** | Snapshots contain all data | Encrypt backups at rest |

---

## Reliability Perspective

### Common Failure Modes and Mitigations

| Failure Mode | Impact | Mitigation |
| --- | --- | --- |
| **Empty results** (filter too strict) | User gets "no results" | Fall back to unfiltered search or broader filter; return "no relevant docs" |
| **Dimension mismatch** | Write fails with cryptic error | Validate dimension at write time; reject mismatched vectors |
| **Embedding model change** | All stored vectors become incompatible | Version collections; re-embed + shadow test before switching |
| **OOM (HNSW in RAM)** | Server crashes | Monitor memory; enable quantization; scale vertically or shard |
| **Stale data after deletes** | Deleted docs still appear in results | Compact/optimize periodically; use stable IDs for idempotent upserts |
| **Slow ingest** | Index falls behind real-time | Batch upserts; async ingest pipeline |

### Idempotent Upserts

Make upserts **idempotent** via stable IDs (hash of chunk content):

```python
import hashlib

def chunk_id(source: str, content: str) -> str:
    """Deterministic ID from content — same content = same ID = idempotent upsert."""
    return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()
```

### Retry Strategy

```python
import time
from qdrant_client.http.exceptions import UnexpectedResponse

def upsert_with_retry(client, collection, points, max_retries=3):
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except UnexpectedResponse as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"Upsert failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
            time.sleep(wait)
```

---

## Evaluation Perspective

### Retrieval Quality Metrics

| Metric | Formula | What It Measures |
| --- | --- | --- |
| **Recall@K** | (relevant in top K) / (total relevant) | Did we find what we should have? |
| **Precision@K** | (relevant in top K) / K | Are returned results relevant? |
| **MRR** | 1 / (rank of first relevant result) | How quickly do we surface the best result? |
| **nDCG@K** | Normalized discounted cumulative gain | Are relevant results ranked higher? |

### Building a Golden Set

```python
# eval/golden.jsonl — 30-50 query-document pairs
GOLDEN = [
    {
        "query": "Why is my pod crashing?",
        "relevant_sources": [
            "kubernetes/crashloopbackoff",
            "kubernetes/oomkilled",
        ],
    },
    {
        "query": "How do I expose my service to the internet?",
        "relevant_sources": ["kubernetes/service-types"],
    },
    {
        "query": "Container image cannot be pulled",
        "relevant_sources": ["kubernetes/imagepullbackoff"],
    },
    {
        "query": "Pod stuck in pending state",
        "relevant_sources": ["kubernetes/pending-pods"],
    },
]
```

### Evaluation Implementation

```python
def evaluate_retrieval(
    search_fn,  # callable: (query, top_k) -> list[{"source": str, "score": float}]
    golden: list[dict],
    top_k: int = 5,
) -> dict:
    """Evaluate retrieval quality against a labeled dataset."""
    total_recall = 0
    total_precision = 0
    total_mrr = 0
    hits = 0

    for case in golden:
        results = search_fn(case["query"], top_k=top_k)
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

    n = len(golden)
    return {
        "recall@k": round(total_recall / n, 3),
        "precision@k": round(total_precision / n, 3),
        "mrr": round(total_mrr / n, 3),
        "hit_rate": round(hits / n, 3),
    }
```

---

## Observability

### What to Monitor

| Metric | Why It Matters | Implementation |
| --- | --- | --- |
| **Search latency (p50/p95/p99)** | Core user-facing performance | Timer per search query |
| **Embedding latency** | Often dominates total latency | Timer per embedding API call |
| **Recall on golden set** | Detect quality degradation | Periodic evaluation run |
| **Collection size** | Capacity planning | Counter from vector DB |
| **Empty result rate** | Detect filter or data issues | Counter per query |
| **Upsert throughput** | Ingest pipeline health | Counter per batch |
| **Memory usage** | Predict OOM before it happens | System metrics from DB |
| **Error rate (search + upsert)** | Detect failures | Error counter |

### Trace the Full Path

```
User query
  → Embedding API call    [latency: 45ms]
  → Vector DB search      [latency: 8ms, results: 5, top score: 0.87]
  → Metadata enrichment   [latency: 2ms]
  → Response formatting   [latency: 1ms]
  → Total                 [56ms]
```

Use **OpenTelemetry** to trace the full retrieve→generate path. Log every query: query text, filters, returned IDs + scores, latency.

---

## Connections

### How Vector Databases Connect to Other AI Engineering Topics

| Topic | Connection |
| --- | --- |
| **Embeddings (Day 4)** | The input to everything here; garbage embeddings = garbage retrieval |
| **RAG (Day 6)** | Vector DB is the "Vector Search" step of the pipeline |
| **Agents (Day 7)** | Retrieval becomes a *tool* the agent calls; vector DB stores agent memory |
| **MCP (Day 8)** | A vector search MCP server exposes retrieval to any AI app |
| **Evaluation (Day 9)** | Retrieval quality metrics (recall, precision) drive system improvement |
| **Chunking** | Chunk size/overlap directly determines what a vector "means" |
| **Hybrid search** | Combine vector (semantic) + BM25 (keyword) for best recall |
| **Re-ranking** | Cross-encoder re-ranks vector DB results for higher precision |

---

## Engineering Tradeoffs

| Advantage | Disadvantage |
| --- | --- |
| Semantic recall with no keyword overlap | Approximate — can miss exact matches (hybrid fixes this) |
| Millisecond search at scale | HNSW is RAM-hungry |
| Metadata filtering for multi-tenancy | Pre/post-filter tradeoffs |
| Incremental inserts (HNSW) | Deletes need compaction |
| Scales horizontally (sharding) | Adds operational complexity |
| Quantization for cost savings | Small recall trade-off |

**When NOT to use a vector database:** if exact keyword/structured lookup suffices, plain SQL/Elasticsearch is simpler and cheaper. Don't reach for vectors when `WHERE name = ?` works.

---

## Implementation

### Python — Qdrant (Complete)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    ScalarQuantizationConfig, ScalarType,
    OptimizersConfigDiff, HnswConfigDiff,
)
from openai import OpenAI
import hashlib

oai = OpenAI()


class QdrantSearchEngine:
    """
    Production-ready vector search engine using Qdrant.

    Handles collection creation, embedding, upsert, and filtered search.
    Uses content-hash IDs for idempotent upserts.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "k8s_docs",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.model = model
        self.dim = dim

    def create_collection(self, quantize: bool = True) -> None:
        """Create collection with HNSW + optional scalar quantization."""
        quantization = (
            ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
            if quantize else None
        )
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
            ),
            quantization_config=quantization,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,  # Build HNSW after 20K vectors
            ),
        )
        print(f"Collection '{self.collection}' created (quantize={quantize})")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI. Batched for efficiency."""
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = oai.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

    @staticmethod
    def chunk_id(source: str, content: str) -> str:
        """Deterministic ID from content for idempotent upserts."""
        return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()

    def index_chunks(self, chunks: list[dict]) -> None:
        """
        Upsert chunks with metadata payload.

        Each chunk: {"text": str, "source": str, "url": str, "heading": str}
        """
        vectors = self.embed([c["text"] for c in chunks])
        points = [
            PointStruct(
                id=self.chunk_id(c["source"], c["text"]),
                vector=v,
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "url": c.get("url", ""),
                    "heading": c.get("heading", ""),
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        # Batch upserts in groups of 100
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i + 100],
            )
        print(f"Indexed {len(chunks)} chunks into '{self.collection}'")

    def search(
        self,
        query: str,
        source: str | None = None,
        k: int = 5,
    ) -> list[dict]:
        """
        Filtered semantic search.

        Returns top-k results with scores and payloads.
        Optionally filter by source (e.g., "kubernetes", "helm").
        """
        query_vec = self.embed([query])[0]

        # Build filter
        flt = None
        if source:
            flt = Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source))
            ])

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            query_filter=flt,
            limit=k,
            with_payload=True,
        )

        return [
            {
                "score": round(h.score, 4),
                "text": h.payload["text"],
                "source": h.payload["source"],
                "url": h.payload.get("url", ""),
                "heading": h.payload.get("heading", ""),
            }
            for h in hits
        ]

    def collection_info(self) -> dict:
        """Get collection stats."""
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }


# Usage
engine = QdrantSearchEngine()
engine.create_collection(quantize=True)

# Index sample chunks
engine.index_chunks([
    {
        "text": "CrashLoopBackOff means the container is repeatedly crashing...",
        "source": "kubernetes",
        "url": "https://k8s.io/docs/troubleshooting",
        "heading": "CrashLoopBackOff",
    },
    # ... more chunks
])

# Search
results = engine.search("why does my pod keep restarting", source="kubernetes")
for r in results:
    print(f"  {r['score']:.4f}  {r['heading']}  ({r['source']})")
```

### SQL — pgvector (Complete)

```sql
-- 1. Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create table with vector column
CREATE TABLE k8s_docs (
    id        bigserial PRIMARY KEY,
    chunk_hash text UNIQUE NOT NULL,    -- for idempotent upserts
    text      text NOT NULL,
    source    text NOT NULL,
    url       text,
    heading   text,
    version   text,
    embedding vector(1536) NOT NULL,
    created_at timestamptz DEFAULT now()
);

-- 3. Create HNSW index with cosine distance
CREATE INDEX idx_k8s_docs_embedding
ON k8s_docs
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- 4. Create payload indexes for filtering
CREATE INDEX idx_k8s_docs_source ON k8s_docs (source);
CREATE INDEX idx_k8s_docs_version ON k8s_docs (version);

-- 5. Set ef_search for queries (session-level)
SET hnsw.ef_search = 128;

-- 6. Filtered nearest-neighbor search (<=> is cosine distance)
SELECT
    id,
    text,
    heading,
    url,
    1 - (embedding <=> $1::vector) AS similarity
FROM k8s_docs
WHERE source = 'kubernetes'
  AND version >= '1.29'
ORDER BY embedding <=> $1::vector
LIMIT 5;

-- 7. Idempotent upsert (ON CONFLICT)
INSERT INTO k8s_docs (chunk_hash, text, source, url, heading, embedding)
VALUES ($1, $2, $3, $4, $5, $6::vector)
ON CONFLICT (chunk_hash)
DO UPDATE SET
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    url = EXCLUDED.url;
```

### Python — pgvector Client

```python
import hashlib
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI

oai = OpenAI()


class PgVectorSearchEngine:
    """
    Vector search engine using pgvector (Postgres extension).

    Simpler ops than a dedicated vector DB — it's just Postgres.
    Best for teams already running Postgres with <5M vectors.
    """

    def __init__(
        self,
        dsn: str = "postgresql://ai:localdev@localhost/embeddings",
        model: str = "text-embedding-3-small",
    ):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        register_vector(self.conn)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI."""
        resp = oai.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def index_chunks(self, chunks: list[dict]) -> None:
        """Upsert chunks with idempotent content-hash IDs."""
        vectors = self.embed([c["text"] for c in chunks])
        cur = self.conn.cursor()
        for chunk, vec in zip(chunks, vectors):
            chunk_hash = hashlib.sha256(
                f"{chunk['source']}:{chunk['text']}".encode()
            ).hexdigest()
            cur.execute(
                """
                INSERT INTO k8s_docs (chunk_hash, text, source, url, heading, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_hash)
                DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                (chunk_hash, chunk["text"], chunk["source"],
                 chunk.get("url", ""), chunk.get("heading", ""), vec),
            )
        print(f"Indexed {len(chunks)} chunks")

    def search(
        self,
        query: str,
        source: str | None = None,
        k: int = 5,
    ) -> list[dict]:
        """Filtered semantic search using pgvector."""
        query_vec = self.embed([query])[0]
        cur = self.conn.cursor()

        sql = """
            SELECT text, source, url, heading,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM k8s_docs
        """
        params = [query_vec]

        if source:
            sql += " WHERE source = %s"
            params.append(source)

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_vec, k])

        cur.execute(sql, params)
        rows = cur.fetchall()

        return [
            {
                "text": row[0],
                "source": row[1],
                "url": row[2],
                "heading": row[3],
                "score": round(float(row[4]), 4),
            }
            for row in rows
        ]
```

### TypeScript — Qdrant

```typescript
import { QdrantClient } from "@qdrant/js-client-rest";
import OpenAI from "openai";

const qdrant = new QdrantClient({ url: "http://localhost:6333" });
const openai = new OpenAI();

const COLLECTION = "k8s_docs";
const DIM = 1536;

async function embed(texts: string[]): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: texts,
  });
  return response.data.map((d) => d.embedding);
}

async function createCollection(): Promise<void> {
  await qdrant.createCollection(COLLECTION, {
    vectors: { size: DIM, distance: "Cosine" },
  });
  console.log(`Collection '${COLLECTION}' created`);
}

interface Chunk {
  text: string;
  source: string;
  url: string;
  heading: string;
}

async function indexChunks(chunks: Chunk[]): Promise<void> {
  const vectors = await embed(chunks.map((c) => c.text));
  const points = chunks.map((chunk, i) => ({
    id: i,
    vector: vectors[i],
    payload: {
      text: chunk.text,
      source: chunk.source,
      url: chunk.url,
      heading: chunk.heading,
    },
  }));
  await qdrant.upsert(COLLECTION, { points });
  console.log(`Indexed ${chunks.length} chunks`);
}

async function search(
  query: string,
  source?: string,
  k: number = 5
): Promise<Array<{ score: number; text: string; source: string }>> {
  const [queryVec] = await embed([query]);

  const filter = source
    ? { must: [{ key: "source", match: { value: source } }] }
    : undefined;

  const results = await qdrant.search(COLLECTION, {
    vector: queryVec,
    filter,
    limit: k,
    with_payload: true,
  });

  return results.map((hit) => ({
    score: hit.score,
    text: (hit.payload as Record<string, string>).text,
    source: (hit.payload as Record<string, string>).source,
  }));
}

// Usage
async function main() {
  await createCollection();
  await indexChunks([
    {
      text: "CrashLoopBackOff means the container is crashing...",
      source: "kubernetes",
      url: "https://k8s.io/docs",
      heading: "CrashLoopBackOff",
    },
  ]);
  const results = await search("pod keeps restarting", "kubernetes");
  console.log(results);
}

main();
```

---

# Building a Kubernetes Documentation Search System

## The Complete Implementation

### Step 1 — Project Setup

```
k8s-doc-search/
├── ingest/
│   ├── crawl.py          # fetch & clean docs (markdown/html)
│   ├── chunk.py          # split into ~500-token overlapping chunks
│   └── index.py          # embed + upsert to Qdrant
├── search/
│   ├── retriever.py      # query embed + filtered ANN
│   └── api.py            # FastAPI /search endpoint
├── eval/
│   └── golden.jsonl      # question → expected source
├── docker-compose.yml    # qdrant + api
├── k8s/                  # deployment + service + qdrant statefulset
└── requirements.txt
```

### Step 2 — Document Chunking

```python
# ingest/chunk.py
import re
from dataclasses import dataclass


@dataclass
class DocChunk:
    """A chunk of documentation with metadata."""
    id: str
    text: str
    source: str       # "kubernetes", "helm", "prometheus"
    url: str
    heading: str
    chunk_index: int


def chunk_markdown(
    text: str,
    source: str,
    url: str = "",
    max_chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocChunk]:
    """
    Chunk Markdown by headers, preserving section structure.
    Each section (## or ###) becomes a chunk.
    Oversized sections are split further with overlap.
    """
    chunks = []
    chunk_index = 0

    # Split by headers
    sections = re.split(r'\n(?=#{1,3} )', text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading
        heading = ""
        if section.startswith("#"):
            heading = section.split("\n")[0].lstrip("# ").strip()

        if len(section) <= max_chunk_size:
            chunks.append(DocChunk(
                id=f"{source}:chunk_{chunk_index}",
                text=section,
                source=source,
                url=url,
                heading=heading,
                chunk_index=chunk_index,
            ))
            chunk_index += 1
        else:
            # Split oversized sections with overlap
            paragraphs = section.split("\n\n")
            current = ""

            for para in paragraphs:
                if len(current) + len(para) > max_chunk_size and current:
                    chunks.append(DocChunk(
                        id=f"{source}:chunk_{chunk_index}",
                        text=current.strip(),
                        source=source,
                        url=url,
                        heading=heading,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
                    # Keep overlap from end of current chunk
                    overlap_text = current[-overlap:] if overlap else ""
                    current = overlap_text + "\n\n" + para
                else:
                    current += ("\n\n" + para if current else para)

            if current.strip():
                chunks.append(DocChunk(
                    id=f"{source}:chunk_{chunk_index}",
                    text=current.strip(),
                    source=source,
                    url=url,
                    heading=heading,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

    return chunks
```

### Step 3 — Embedding and Indexing

```python
# ingest/index.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    ScalarQuantizationConfig, ScalarType,
)
from openai import OpenAI
from chunk import chunk_markdown, DocChunk
import hashlib

oai = OpenAI()
qc = QdrantClient(url="http://localhost:6333")

COLLECTION = "k8s_docs"
DIM = 1536
MODEL = "text-embedding-3-small"


def setup_collection():
    """Create collection with HNSW + scalar quantization."""
    qc.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        quantization_config=ScalarQuantizationConfig(
            type=ScalarType.INT8, quantile=0.99, always_ram=True,
        ),
    )


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = oai.embeddings.create(model=MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return embeddings


def index_documents(docs: list[dict]):
    """
    Index documents: chunk → embed → upsert.

    Each doc: {"content": str, "source": str, "url": str}
    """
    # Chunk all documents
    all_chunks = []
    for doc in docs:
        chunks = chunk_markdown(
            text=doc["content"],
            source=doc["source"],
            url=doc.get("url", ""),
        )
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")

    # Embed
    vectors = embed_batch([c.text for c in all_chunks])

    # Upsert
    points = [
        PointStruct(
            id=hashlib.sha256(f"{c.source}:{c.text}".encode()).hexdigest(),
            vector=v,
            payload={
                "text": c.text,
                "source": c.source,
                "url": c.url,
                "heading": c.heading,
            },
        )
        for c, v in zip(all_chunks, vectors)
    ]

    for i in range(0, len(points), 100):
        qc.upsert(collection_name=COLLECTION, points=points[i:i + 100])

    print(f"Indexed {len(points)} chunks into '{COLLECTION}'")


if __name__ == "__main__":
    setup_collection()
    # Load your docs here
    # index_documents(docs)
```

### Step 4 — Search API

```python
# search/api.py
from fastapi import FastAPI, Query
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

app = FastAPI(title="K8s Doc Search")
qc = QdrantClient(url="http://qdrant:6333")
oai = OpenAI()

COLLECTION = "k8s_docs"
MODEL = "text-embedding-3-small"


@app.get("/search")
async def search(
    q: str = Query(..., description="Natural-language search query"),
    source: str | None = Query(None, description="Filter by source: kubernetes, helm, prometheus"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
):
    """Semantic search over Kubernetes documentation."""
    # Embed query
    resp = oai.embeddings.create(model=MODEL, input=[q])
    query_vec = resp.data[0].embedding

    # Build filter
    flt = None
    if source:
        flt = Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=source))
        ])

    # Search
    hits = qc.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        query_filter=flt,
        limit=k,
        with_payload=True,
    )

    return {
        "query": q,
        "results": [
            {
                "score": round(h.score, 4),
                "text": h.payload["text"],
                "source": h.payload["source"],
                "url": h.payload.get("url", ""),
                "heading": h.payload.get("heading", ""),
            }
            for h in hits
        ],
    }


@app.get("/health")
async def health():
    info = qc.get_collection(COLLECTION)
    return {
        "status": "ok",
        "vectors": info.vectors_count,
        "indexed": info.indexed_vectors_count,
    }
```

### Step 5 — Evaluation

```python
# eval/run_eval.py
import json
from search.retriever import search

GOLDEN = json.load(open("eval/golden.jsonl"))


def evaluate(top_k: int = 5) -> dict:
    """Evaluate retrieval quality against golden set."""
    total_recall = 0
    total_mrr = 0
    hits = 0

    for case in GOLDEN:
        results = search(case["query"], top_k=top_k)
        result_sources = [r["source"] for r in results]
        relevant = set(case["relevant_sources"])

        found = sum(1 for s in result_sources if s in relevant)
        total_recall += found / len(relevant) if relevant else 0

        for rank, source in enumerate(result_sources, 1):
            if source in relevant:
                total_mrr += 1 / rank
                break

        if found > 0:
            hits += 1

    n = len(GOLDEN)
    metrics = {
        "recall@k": round(total_recall / n, 3),
        "mrr": round(total_mrr / n, 3),
        "hit_rate": round(hits / n, 3),
        "cases": n,
        "top_k": top_k,
    }
    print(json.dumps(metrics, indent=2))

    # CI gate
    assert metrics["recall@k"] >= 0.85, f"Recall regression: {metrics['recall@k']}"
    return metrics


if __name__ == "__main__":
    evaluate()
```

---

## Production Architecture

Publicly known pattern (e.g., Perplexity/Glean-style retrieval): documents are chunked and embedded in an **async ingest pipeline** (queue + workers), written to a **sharded vector store**, and served by a **stateless retrieval service** that does query embedding (with caching) + ANN + filtering, often followed by a **re-ranker** (Day 6). This is an architectural pattern, not a claim about any proprietary internals.

```
  Document Sources          Async Ingest Pipeline           Vector Store
  ┌──────────┐             ┌──────────┐                    ┌──────────┐
  │  Git     │──┐          │  Queue   │                    │  Qdrant  │
  │  Wiki    │──┼──────►   │  (SQS/   │──► Workers ──────►│  (sharded│
  │  Docs    │──┤          │  Kafka)  │    (chunk+embed)   │  +quant) │
  │  APIs    │──┘          └──────────┘                    └────┬─────┘
                                                                │
  User Query ──► Query Embedding ──► Vector Search ──► Re-rank ──► LLM
                  (cached)           (ANN + filter)    (Day 6)    (Day 6)
```

---

## Expert Insights

### What Beginners Think

- "Vector DB = magic search box; just embed and query."
- "All vector databases are the same; just pick one."
- "I don't need an index for 100K vectors." (You don't — but you will at 1M.)
- "Filtering is easy — just post-filter." (Post-filtering can return < k results.)

### What Experienced Engineers Know

- Retrieval quality is mostly about **chunking + the right metric + hybrid search + filtering**, not the DB brand.
- **HNSW parameter tuning** (M, ef_construction, ef_search) has more impact on recall/latency than switching databases.
- **Quantization** should be enabled from the start, not bolted on when you run out of RAM.
- **Metadata filtering** is as important as vector search — it enables tenancy, versioning, and precision.
- The **embedding API call** often dominates total latency, not the vector search itself.

### What Staff Engineers Optimize For

- Recall@k *and* p99 latency *and* RAM cost simultaneously via quantization and index tuning.
- Multi-tenant isolation with mandatory server-side filters.
- Incremental indexing with content hashing to avoid re-embedding unchanged documents.
- Capacity planning: predicting when you'll need to shard or upgrade memory.
- Shadow-testing new embedding models before switching.

### What Frontier Labs Care About

- Billion-scale ANN with sub-50ms p99.
- Freshness: real-time upserts reflected in search within seconds.
- Retrieval evals wired into CI — every change to chunking, model, or index params triggers a recall benchmark.
- Hybrid sparse-dense search in a single index.

### Common Interview Trap

Confusing **exact vs. approximate** nearest neighbor, or picking the **wrong distance metric** for the model. Also: not knowing what `ef_search` does or how HNSW works at a high level.

### Production Failure Modes

- Dimension mismatch (silently bad results or hard errors)
- Over-strict filters returning nothing
- Stale index after deletes (tombstones)
- RAM OOM from un-quantized large collections
- Mixing vectors from different embedding models in one collection

---

## Evolution of the Field

```
2014: LSH (Locality-Sensitive Hashing) — early ANN for large-scale similarity
  ↓
2015: Annoy (Spotify) — tree-based ANN for recommendations
  ↓
2017: FAISS (Facebook) — GPU-accelerated IVF/PQ, gold standard for research
  ↓
2018: HNSW paper (Malkov & Yashunin) — graph-based ANN, best recall/latency
  ↓
2019: Milvus, Pinecone — first dedicated vector database products
  ↓
2020: Weaviate — vector DB with built-in embedding + hybrid search
  ↓
2021: Qdrant — Rust-based, filterable HNSW, K8s-native
       pgvector — Postgres extension for vectors
  ↓
2022: Pinecone raises, Chroma for prototyping, vector DB hype cycle
  ↓
2023: pgvector adds HNSW, Qdrant adds quantization
       Vector databases become standard infrastructure
  ↓
2024: Binary/scalar quantization mainstream, DiskANN for SSD-based search
       Hybrid sparse-dense search in dedicated DBs
  ↓
2025: Billion-scale with real-time freshness, learned indexes, multi-vector
       (ColBERT-style late interaction), auto-tuning index parameters
  ↓
Emerging: Learned indexes, multi-modal vector search (text + image + code),
          serverless vector databases, auto-quantization
```

### Established Practice

- HNSW as the default ANN algorithm
- pgvector for simple use cases, Qdrant/Milvus for dedicated workloads
- Scalar quantization for memory savings
- Metadata filtering for multi-tenancy
- Content-hash IDs for idempotent upserts

### Emerging Practice

- Binary quantization for 32x compression
- DiskANN for SSD-based billion-scale search
- Hybrid sparse-dense indexes in a single collection
- Matryoshka embeddings for flexible dimension sizing
- Auto-tuning HNSW parameters based on workload

### Research Direction

- Can we build vector indexes that are exact AND fast? (learned indexes)
- How do we handle real-time freshness at billion scale?
- Multi-modal vector spaces (text + image + code in one index)
- Verifiable/auditable nearest-neighbor search

---

## Hands-On Project — Kubernetes Documentation Search

### Problem

Ingest Kubernetes, Helm, and Prometheus docs; answer natural-language queries with cited source chunks.

### Architecture

```
Markdown Docs (K8s + Helm + Prometheus)
    │
    ▼
┌──────────────────────┐
│  Markdown Chunker     │  Split by headers, 500-token chunks, 15% overlap
├──────────────────────┤
│  Embedding API        │  text-embedding-3-small (1536 dims)
├──────────────────────┤
│  Qdrant (quantized)   │  HNSW + scalar quantization, payload indexes
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │  Collection  │  "k8s_docs" with source/url/heading payloads
    └──────┬──────┘
           │
    User Query
    │
    ▼
┌──────────────────────┐
│  FastAPI /search      │  Embed query → filtered ANN → top-k results
├──────────────────────┤
│  Result Formatter     │  Score, source, heading, text preview, URL
└──────────────────────┘
    │
    ▼
  Ranked Results with Citations
```

### Implementation Steps

1. Set up Qdrant with docker-compose
2. Implement Markdown-aware chunker (split on headers + code fences)
3. Embed chunks with `text-embedding-3-small`
4. Upsert to Qdrant with `source`, `url`, `heading` in payload
5. Build `/search?q=&source=` API returning top-k with URLs
6. Build golden set of ~30 K8s questions with known-correct source chunks
7. Run retrieval eval, ensure recall@5 ≥ 0.85

### Production Improvements

- Add **hybrid search** (BM25 + vector) with Reciprocal Rank Fusion
- Add a **cross-encoder re-ranker** (Day 6)
- Add **query-embedding cache** (Redis)
- Enable **scalar quantization**
- Add **incremental indexing** with content hashing

### Scaling

- Shard Qdrant collection across nodes
- Async ingest via a queue (SQS/Kafka)
- Horizontal-scale the stateless search API

### Deploy on Kubernetes

- Qdrant as a StatefulSet with PVCs
- Search API as a Deployment + Service + HPA
- Embedding API key via Kubernetes Secret

### Monitoring

- Search latency p50/p95/p99
- Recall on golden set (periodic eval)
- Empty-result rate, collection size, memory usage
- Embedding API latency and error rate

### Testing

- Unit tests: chunking logic, ID generation
- Integration tests: embed + upsert + search against local Qdrant
- Eval tests: run `eval/golden.jsonl` in CI, fail build if recall@5 drops

---

## Exercises

### Beginner Exercises

1. Create a Qdrant collection, upsert 100 chunks with metadata, and run a semantic search.
2. Create a pgvector table, insert 100 vectors, and query the 5 nearest neighbors.
3. Compute cosine similarity manually (NumPy) between two embedding vectors and verify it matches the vector DB score.
4. Experiment with different queries for the same concept ("pod crashing" vs "container restart" vs "CrashLoopBackOff") — do they find the same results?
5. Save and load a collection snapshot in Qdrant.

### Intermediate Exercises

1. Add metadata filtering by `source` and `version`; verify filtered results are correct.
2. Measure recall@5 on a 20-question golden set against both Qdrant and pgvector.
3. Implement idempotent upserts with content-hash IDs; verify re-ingesting the same data doesn't create duplicates.
4. Build a FastAPI search endpoint with query, filter, and top-k parameters.
5. Compare search results from cosine vs. dot product vs. Euclidean distance metrics.

### Advanced Exercises

1. Compare HNSW vs IVFFlat on recall vs latency vs RAM at 100K and 1M vectors.
2. Tune `ef_search` from 50 to 512 and plot the recall/latency curve.
3. Enable scalar quantization and measure the recall/memory tradeoff.
4. Implement incremental indexing with content hashing — only re-embed changed documents.
5. Add hybrid search (BM25 + vector) with Reciprocal Rank Fusion and measure the recall improvement.

### Production Engineering Exercises

1. Design a multi-tenant vector search system — each tenant's data isolated via mandatory server-side filters.
2. Build a monitoring dashboard tracking search latency (p95), recall, collection size, and memory usage.
3. Implement a migration strategy for switching embedding models without downtime.
4. Set up Qdrant as a StatefulSet on Kubernetes with PVCs, readiness/liveness probes, and resource limits.
5. Implement a query-embedding cache with Redis to avoid redundant embedding API calls.

### System Design Exercises

1. **Design a 1-billion-vector search service** with <50ms p99 and hourly freshness. Consider: sharding, replication, quantization, ingest pipeline, capacity planning.
2. **Design the vector search layer for an enterprise RAG platform** serving 100 tenants with 10M vectors each. Consider: tenant isolation, filtering, scaling, cost.
3. **Design a real-time vector search system** that reflects document updates in search results within 5 seconds. Consider: streaming ingest, index refresh, consistency.

---

## Key Takeaways

1. **Vector DBs answer "nearest in meaning," not "matching keywords."** They find semantically similar items by computing distances between high-dimensional embedding vectors.
2. **Embeddings turn meaning into geometry; search = nearest-neighbor in that geometry.** Small distance = similar meaning. The vector database makes this search fast.
3. **One embedding model per collection** — never mix vector spaces. Switching models requires re-embedding everything.
4. **HNSW is the modern default index.** Tune `M`/`ef_construction` at build time and `ef_search` at query time to trade recall ↔ latency. Start with M=16, ef_construction=128.
5. **Pick the distance metric your model expects** (usually cosine for text). Wrong metric = silently terrible recall.
6. **Metadata + filtering are as important as the vectors** — they enable tenancy, versioning, scoping, and citations.
7. **Measure recall@k AND p95 latency together** — neither alone is meaningful. Aim for ≥0.95 recall in production.
8. **Quantization trades a little recall for large RAM/cost savings.** Scalar quantization (int8) gives 4x savings with ~1% recall loss — enable it from the start.
9. **pgvector for simplicity, Qdrant for a first-class filterable retrieval workload.** Both are excellent defaults; avoid Milvus unless you truly need billion-scale.
10. **Enforce tenant isolation with server-side mandatory filters.** Never trust client-side filtering for multi-tenant security.
11. **Retrieval quality usually beats model size for answer quality.** Invest in better retrieval before reaching for a bigger LLM.
12. **Make upserts idempotent with stable content-hash IDs.** Same content = same ID = safe to re-ingest.
13. **Handle empty results gracefully** — fall back to unfiltered search or return "no relevant docs."
14. **Wire retrieval evals into CI** so regressions in recall are caught automatically before they ship.
15. **The embedding API call often dominates total latency** — batch embeddings and cache query embeddings.
16. **ANN is approximate** — it can miss some true nearest neighbors. Hybrid search (vector + BM25) recovers exact keyword matches that ANN might miss.
17. **Capacity plan proactively**: estimate memory as N × D × 4 bytes × ~1.7x overhead. Know when you'll need to shard or quantize.
