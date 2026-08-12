# semantic-engine

[![CI](https://github.com/aaravgarai17/semantic-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aaravgarai17/semantic-engine/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented question answering system built like infrastructure
rather than a demo: **hybrid retrieval** (dense vectors + BM25 fused with
Reciprocal Rank Fusion), asynchronous ingestion, answers with resolvable
citations, and an evaluation harness that measures whether any of it actually
works.

Most RAG projects are a script and a vector database. The system design here
matters as much as the AI — and every claim is **measured, not asserted**.

The measurement produced a result I did not expect: on this corpus, **hybrid
retrieval performs worse than dense retrieval alone**. That finding, and the
mechanism behind it, is [the most interesting thing in this
repository](#the-headline-result--hybrid-retrieval-lost).

---

## Verify it in one command

> **Common tasks:** `make help` lists everything — `make install`, `make test`,
> `make verify`, and per-project shortcuts.

```bash
pip install -r requirements.txt
./verify.sh
```

Runs entirely offline. Checks BM25 identifier retrieval, RRF's fusion property,
structure-aware chunking, and runs the full retrieval evaluation.

```
 ✓ all tests passed
 ✓ exact identifier retrieved, rare terms weighted above common ones
 ✓ agreement across retrievers outranks a single first place
 ✓ heading trail tracked and prepended to embedded text
 ✓ 40 golden questions across paraphrase, identifier and mixed types
 ✓ all three retrieval strategies scored
 ✓ evaluation reports a verdict on whether fusion helped

 Results: 8 passed, 0 failed
```

Note what this deliberately does **not** assert: that hybrid wins. Whether
fusion helps is the thing being measured, and gating CI on a particular outcome
would turn a measurement into a target — the first time a legitimate run
disagreed, the temptation would be to adjust the corpus until it agreed.

---

## The headline result — hybrid retrieval *lost*

The question this project exists to answer is whether combining retrieval
methods beats using one. I built the evaluation expecting to demonstrate that
it does. **It doesn't, on this corpus** — and understanding why turned out to
be more interesting than the result I was expecting.

Reproduce it:

```bash
pip install sentence-transformers
python -m eval.run_eval --embedder local --examples
```

14 documents → 49 chunks, **40 hand-labelled questions**,
`all-MiniLM-L6-v2` embeddings, Apple M2.

| Method | recall@5 | recall@10 | MRR | latency |
| ------ | -------- | --------- | --- | ------- |
| **dense only** | **97.5%** | 100.0% | **0.806** | 1.29 ms |
| BM25 only | 62.5% | 65.0% | 0.536 | 0.03 ms |
| hybrid (RRF) | 92.5% | 100.0% | 0.729 | 1.34 ms |

Dense alone beat the fusion by **5.0 points** at recall@5.

The result held across two independently constructed question sets — an
earlier 22-question version showed −4.5 points, and expanding to 40 questions
with harder out-of-vocabulary identifiers moved it to −5.0. It is not an
artifact of one unlucky sample.

### The precise failure: ranking, not recall

Look at recall@10 — **dense and hybrid are tied at 100%**. Fusion finds
everything dense finds. What it loses is *ordering within the top 5*.

That matters practically: a generator given the top 5 chunks gets a worse
context set from hybrid, even though both retrievers would have surfaced the
right passage eventually. Recall@10 hides the problem entirely, which is why
reporting a single k is misleading.

### Why — and it is not that RRF is broken

| Question type | dense | BM25 | hybrid |
| ------------- | ----- | ---- | ------ |
| paraphrase (18 q) — no shared vocabulary | **94.4%** | 27.8% | 83.3% |
| identifier (12 q) — exact codes and paths | 100.0% | 91.7% | 100.0% |
| mixed (10 q) — both signals | 100.0% | 90.0% | 100.0% |

The damage is entirely in the paraphrase row. BM25 scores **27.8%** there — it
is returning close to noise — and equal-weight RRF injects that noise straight
into the top ranks, pulling hybrid from 94.4% down to 83.3%.

Where the retrievers were comparable, fusion did exactly what it promises. On
identifier questions hybrid MRR is **0.938**, beating both dense (0.903) and
BM25 (0.917) — both lists agreed on the right chunk, and agreement is what RRF
rewards.

**The prediction I got wrong.** I expected embeddings to smear `ERR_4032` into
generic "error" space, making identifiers the category BM25 wins. They didn't —
dense scored **100%** on identifiers. A modern sentence-transformer retains
enough lexical signal that an exact-token query still matches the chunk
containing that token.

I tested this deliberately by adding identifiers a model is unlikely to have
seen in training — `libpq-dev`, `BLMOVE`, `vector_cosine_ops`, `ts_rank_cd`.
Dense still handled them. The "embeddings can't do exact match" intuition is
accurate for older models and largely obsolete for current ones.

**The actual mechanism.** Equal-weight fusion is only safe when the retrievers
are comparably good. Fusion didn't fail — it faithfully averaged a strong
ranking with a weak one, which is what it was asked to do.

### The fix, and what it costs

```bash
python -m eval.run_eval --embedder local --weight-sweep
```

Weighting the fusion toward the stronger retriever recovers most of the loss.
But note what that means: **the weights have to be tuned per corpus**, and
tuning needs a labelled set. Plain RRF's whole appeal was working without one.
Weighted RRF trades that away.

### What I'd actually conclude

Hybrid retrieval is not free insurance. It helps when neither retriever
dominates — a corpus with genuinely out-of-vocabulary identifiers, product SKUs,
or a domain the embedding model never saw. On a small corpus of clean prose
that a good embedding model handles well, dense alone is better, faster, and
simpler.

The honest recommendation from this evaluation is: **measure before adding
fusion.** The complexity is only worth it if your corpus has the properties
that make it pay.

> A note on the corpus: 49 chunks of clean synthetic prose is small and easy.
> A larger corpus with real out-of-vocabulary terms would likely favour hybrid
> more. That is a limitation of the evaluation, listed with the others below —
> not a reason to discard its result.

---

## The theory behind hybrid retrieval

This is the reasoning the system was built on. The evaluation above tests it,
and partly refutes it — worth reading in that order.

### Dense search matches meaning

Embeddings place semantically similar text near each other, so *"how do I reset
my password"* retrieves *"credential recovery requires identity verification"*
despite sharing **no words at all**. Measured: 100% recall@5 on paraphrase
questions, against BM25's 27.8%. This part held up completely.

### BM25 matches words

BM25 scores by term overlap, so `ERR_4032` finds exactly the passage containing
that literal string, and it has no notion of synonymy — *"password reset"* and
*"credential recovery"* share nothing.

**The predicted complementarity did not materialise.** The theory says dense
should fail on identifiers because an error code has no semantic content to
embed. Measured, dense scored 100% there too. Modern sentence-transformers
retain more lexical signal than the "embeddings can't do exact match" framing
suggests — that intuition is accurate for older models and largely stale for
current ones.

BM25 still earns its place: it is **40× faster** (0.03 ms vs 1.14 ms), needs no
model, and would come into its own on a corpus full of terms the embedding
model never saw — internal product codes, part numbers, a specialised domain.
Just not this one.

### Why fusing scores doesn't work, and RRF does

The obvious combination — normalise both scores and add them — has a real
problem: **the scores are not commensurable**. A cosine similarity of 0.83 and
a BM25 score of 14.2 live on different scales with different distributions.
BM25 is unbounded above and depends on corpus statistics; cosine is bounded in
[-1, 1] and clusters tightly, so the spread that matters is small and easily
swamped. Min-max normalising per query makes a document's score depend on which
*other* documents happened to be retrieved alongside it.

Reciprocal Rank Fusion discards the scores entirely and uses only rank:

```
RRF(d) = Σ  1 / (k + rank_i(d))
         i
```

Documents ranked highly by **both** methods accumulate the most weight. A
document one method loves and the other has never seen still scores
respectably — which is exactly the case where one retriever is covering the
other's blind spot.

**Why k = 60.** The constant damps the advantage of top ranks. Without it,
rank 1 scores 1.0 and rank 2 scores 0.5, so a single first place is nearly
impossible to overcome. With k=60 they score 1/61 and 1/62 — close enough that
agreement across retrievers outweighs being first in one. There's a test
demonstrating exactly this inversion.

RRF needs no tuning, no training data, and no score calibration. That is the
argument for it over a learned reranker in a system this size.

---

## Architecture

```
  upload ──▶ ┌──────────────────┐
             │  POST /documents │──▶ 202 Accepted + job id
             └────────┬─────────┘         (work has not started)
                      ▼  background
   ┌────────────────────────────────────────────────────┐
   │  extract (PDF/DOCX/MD/TXT, page numbers preserved) │
   │  chunk   (headings → paragraphs → sentences)       │
   │  embed   (batched, never one call per chunk)       │
   │  store   (vectors + keyword index)                 │
   └────────────────────────────────────────────────────┘
                      │
   query ──▶ ┌────────┴─────────┐
             │  dense search    │──┐
             │  BM25 search     │──┼──▶ RRF fusion ──▶ top-k chunks
             └──────────────────┘  │
                                   ▼
             ┌──────────────────────────────────────┐
             │  assemble numbered context           │
             │  generate answer                     │
             │  parse citations, drop fabrications  │
             │  refuse if nothing relevant          │
             └──────────────────────────────────────┘
```

---

## Chunking: structure before size

Fixed-size chunking splits sentences in half, separates tables from headers,
and detaches a heading from the section it introduces. The consequence is
specific: a chunk severed mid-sentence embeds as something close to noise, so
it matches nothing, and whatever it contained becomes unreachable.

The strategy falls back progressively:

1. **Headings** — markdown, setext, and numbered sections. A heading marks a
   topic change, which is exactly a chunk boundary.
2. **Paragraphs**, inside an oversized section.
3. **Sentences**, inside an oversized paragraph.
4. **Hard cut**, only if a single sentence exceeds the budget.

**Heading trails are carried and embedded.** A chunk under
`Troubleshooting > Network > DNS` often never repeats those words, making it
unreachable by a query about DNS troubleshooting. Prepending the trail to the
embedded text costs a few tokens and measurably improves recall. Citations get
it for free.

**Overlap** — adjacent chunks share a tail, cut at a word boundary. Without it,
a fact spanning a boundary is retrievable from neither side.

Run the chunk-size experiment yourself:

```bash
python -m eval.run_eval --sweep
```

---

## Citations, and refusing to answer

An answer without citations is indistinguishable from a hallucination. Excerpts
are numbered in the prompt, the model must cite by number, and citations are
parsed back out and **resolved against what was actually shown** — a citation
to `[9]` when five excerpts were provided is a fabrication and gets dropped.

**The system must be able to say "I don't know."** A model given irrelevant
context will still answer, confidently, from parametric memory — the worst
failure mode for a document-grounded system, because it looks exactly like
success. Two defences:

- a **retrieval score floor** (`MIN_RETRIEVAL_SCORE`) that refuses before
  spending a request — cheaper and more reliable than hoping the model declines
- an explicit instruction that refusing is *correct*, not a failure

An uncited, non-refusing answer is flagged as ungrounded rather than returned
silently.

---

## Ingestion is asynchronous, and that's not optional

A 300-page PDF takes tens of seconds to extract, chunk, and embed. Doing that
inside the upload request holds the connection open, trips every proxy timeout
in between, and leaves the user with no idea whether it's working.

So upload returns **202 Accepted** with a job id, and status is polled:

```json
{"status": "embedding", "progress": 62, "chunks": 847, "pages": 300}
```

Progress is weighted by stage, with embedding dominating — a progress bar that
sits at 90% through the slowest stage is worse than none. Concurrency is
bounded by a semaphore: fifty simultaneous ingestions would exhaust memory or
hit a provider rate limit, turning a slow ingest into a failed one.

---

## Running it

**Requires:** Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended — real semantic embeddings, offline, no key
pip install sentence-transformers

cp .env.example .env
uvicorn app.main:app --reload
```

```bash
# Ingest
curl -F "file=@manual.pdf" -F "collection=docs" http://localhost:8000/documents
# → 202 {"job_id": "a1b2c3", "poll": "/documents/a1b2c3"}

curl http://localhost:8000/documents/a1b2c3
# → {"status": "embedding", "progress": 62, ...}

# Query — retrieval only, no API key needed
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "what does ERR_4032 mean?", "generate": false}'

# Compare retrieval methods on the same question
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "how do I reset my password", "method": "dense",  "generate": false}'
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "how do I reset my password", "method": "bm25",   "generate": false}'
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "how do I reset my password", "method": "hybrid", "generate": false}'
```

Set `ANTHROPIC_API_KEY` to get generated answers with citations. Without it,
retrieval works and the response says so.

Results carry provenance — `dense_rank`, `bm25_rank`, and `found_by_both` —
so fusion is inspectable rather than a black box.

With Docker:

```bash
docker compose up --build
```

## Tests

```bash
pytest -q                                     # 131 tests
pytest --cov=app --cov-report=term-missing    # 90%
```

Everything runs offline via the non-semantic hash embedder. Covers BM25
scoring properties (IDF weighting, term-frequency saturation, length
normalisation), RRF's fusion behaviour, chunking across markdown/setext/
numbered headings, citation parsing including fabricated references, ingestion
including failure paths, and the API end to end.

CI runs on Python 3.10/3.11/3.12, executes the retrieval evaluation (which
**exits non-zero if hybrid fails to beat the best single method**), runs an API
smoke test, and on `main` re-runs the evaluation with real sentence-transformer
embeddings.

---

## What doesn't work well

- **No reranking.** A cross-encoder over the top ~20 fused candidates would
  likely add several points of precision. It's the highest-value next step and
  it isn't here.
- **In-memory store is the default.** Exact and dependency-free, but the whole
  corpus lives in one process. `PgVectorStore` has the schema and HNSW index
  defined; the async query path isn't wired up.
- **Postgres keyword search would differ.** The in-memory path uses the BM25
  written here; pgvector's path uses `ts_rank_cd`, a different scoring
  function. Results would shift slightly between backends.
- **The evaluation corpus is synthetic, small, and easy.** 13 documents, 45
  chunks, 22 questions, written by me. It is clean prose on distinct topics —
  close to the best case for an embedding model, which is a large part of why
  dense retrieval scored 100% and fusion had nothing to add. A corpus with real
  out-of-vocabulary identifiers, near-duplicate passages, and messier writing
  would likely favour hybrid more. **The headline finding should be read as
  "hybrid didn't help *here*", not "hybrid doesn't help".** A public benchmark
  (BEIR, MS MARCO) would settle it properly and is the honest next step.
- **Only one embedding model was tested.** `all-MiniLM-L6-v2`. A weaker model
  would plausibly flip the result, since fusion helps most when the dense
  retriever is mediocre.
- **No semantic cache or multi-turn rewriting.** Phase 6 of the plan; v1 ships
  without them.
- **No reciprocal-rank tuning.** k=60 is the conventional default, not a value
  fitted to this corpus.
- **`all-MiniLM-L6-v2` is a small model.** Chosen so the evaluation runs offline
  and free. A hosted embedder would score better; the *relative* comparison
  between methods is what this measures.
- **English only**, and sentence splitting is regex-based rather than a real
  parser.

## Layout

```
semantic-engine/
├── app/
│   ├── retrieval.py   # BM25, dense search, RRF        ← the core
│   ├── chunking.py    # structure-aware splitting
│   ├── embeddings.py  # pluggable providers
│   ├── store.py       # in-memory + pgvector schema
│   ├── ingest.py      # async pipeline with status
│   ├── generate.py    # citations and refusal
│   ├── extract.py     # PDF / DOCX / MD / TXT
│   └── main.py        # FastAPI
├── eval/
│   ├── dataset.py     # 13 docs, 22 labelled questions
│   └── run_eval.py    # the comparison table
├── tests/             # 131 tests, fully offline
└── verify.sh          # 7 checks proving this README
```
