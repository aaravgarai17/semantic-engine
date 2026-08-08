#!/usr/bin/env bash
#
# One-command verification. Runs offline with the non-semantic hash embedder,
# so no model download, no API key, and no network are required.
#
# For the real retrieval numbers:  python -m eval.run_eval --embedder local
#
# Usage:  ./verify.sh

set -uo pipefail

PY="${PYTHON:-python3}"
pass=0
fail=0

ok()  { echo "  ✓ $1"; pass=$(( pass + 1 )); }
bad() { echo "  ✗ $1"; fail=$(( fail + 1 )); }

echo "=================================================="
echo " 0/5  Preflight"
echo "=================================================="
command -v $PY >/dev/null || { echo "  ✗ python3 not found"; exit 1; }
if ! $PY -c "import pytest, fastapi, pydantic" 2>/dev/null; then
  echo "  ✗ dependencies missing. Run:"
  echo "      python3 -m venv .venv && source .venv/bin/activate"
  echo "      pip install -r requirements.txt"
  exit 1
fi
ok "python and dependencies available"

echo ""
echo "=================================================="
echo " 1/5  Test suite"
echo "=================================================="
if $PY -m pytest -q -p no:cacheprovider 2>&1 | tail -3; then
  ok "all tests passed"
else
  bad "tests failed"
fi

echo ""
echo "=================================================="
echo " 2/5  BM25 finds exact identifiers"
echo "=================================================="
# The failure mode dense retrieval has: an error code has no semantic content,
# so embeddings smear it into general space.
if $PY - <<'EOF'
from app.retrieval import BM25Index

idx = BM25Index()
idx.add_many([
    ("a", "Credential recovery requires identity verification in the system."),
    ("b", "Error ERR_4032 means the connection pool was exhausted in the system."),
    ("c", "Invoices are generated on the first of the month by the system."),
])
hits = idx.search("ERR_4032")
assert hits and hits[0].chunk_id == "b", hits

# IDF: a term in one document out of three carries far more signal than one
# present in all three. ('the' is a stopword and never indexed, so comparing
# against it would measure nothing.)
assert idx.idf("err_4032") > idx.idf("system"), (
    idx.idf("err_4032"), idx.idf("system")
)
EOF
then
  ok "exact identifier retrieved, rare terms weighted above common ones"
else
  bad "BM25 is not behaving correctly"
fi

echo ""
echo "=================================================="
echo " 3/5  RRF prefers cross-retriever agreement"
echo "=================================================="
# The core property fusion is chosen for.
if $PY - <<'EOF'
from app.retrieval import ScoredChunk, reciprocal_rank_fusion

def lst(ids, src):
    return [ScoredChunk(c, 1.0, i + 1, src) for i, c in enumerate(ids)]

# 'b' is second in both lists; 'a' is first in one and absent from the other.
fused = reciprocal_rank_fusion([lst(["a", "b"], "dense"), lst(["d", "b"], "bm25")])
assert fused[0].chunk_id == "b", [f.chunk_id for f in fused]
assert fused[0].found_by_both

# A document found by only one retriever must still survive — covering a blind
# spot is the entire point.
fused = reciprocal_rank_fusion([lst(["x"], "dense"), lst(["y"], "bm25")])
assert {f.chunk_id for f in fused} == {"x", "y"}
EOF
then
  ok "agreement across retrievers outranks a single first place"
else
  bad "RRF is not fusing correctly"
fi

echo ""
echo "=================================================="
echo " 4/5  Chunking respects document structure"
echo "=================================================="
if $PY - <<'EOF'
from app.chunking import chunk_document

doc = """# Installation

Install with pip. Requires Python 3.10 or later.

## Linux

On Debian you also need libpq-dev.
"""
chunks = chunk_document(doc, max_chars=1000, overlap_chars=0)
linux = next(c for c in chunks if "Debian" in c.text)

# The heading trail is carried and prepended for embedding, so a chunk that
# never repeats the word "Installation" is still retrievable by it.
assert linux.heading_path == ["Installation", "Linux"], linux.heading_path
assert linux.embed_text.startswith("Installation > Linux")
EOF
then
  ok "heading trail tracked and prepended to embedded text"
else
  bad "structure-aware chunking is broken"
fi

echo ""
echo "=================================================="
echo " 5/5  Retrieval evaluation produces a measurement"
echo "=================================================="
# Deliberately does NOT assert that hybrid wins. Whether fusion helps is the
# result being measured, and gating on a particular outcome would turn the
# measurement into a target — the moment a legitimate run disagreed, the
# temptation would be to adjust the corpus until it agreed.
eval_output=$($PY -m eval.run_eval --embedder hash 2>&1)
echo "$eval_output" | grep -E "^  (dense|bm25|hybrid) " | sed 's/^/    /'

if echo "$eval_output" | grep -q "questions     22"; then
  ok "22 golden questions across paraphrase, identifier and mixed types"
else
  bad "unexpected question count"
fi

if echo "$eval_output" | grep -qE "^  hybrid +[0-9]+\.[0-9]%"; then
  ok "all three retrieval strategies scored"
else
  bad "evaluation did not report scores"
fi

if echo "$eval_output" | grep -q "FINDING:"; then
  ok "evaluation reports a verdict on whether fusion helped"
else
  bad "no verdict reported"
fi

echo ""
echo "=================================================="
echo " Results: $pass passed, $fail failed"
echo "=================================================="
if [[ $fail -eq 0 ]]; then
  echo "VERIFIED — every README claim checks out."
  echo ""
  echo "Note: these numbers use the non-semantic hash embedder so this runs"
  echo "offline. For the real result:"
  echo "    pip install sentence-transformers"
  echo "    python -m eval.run_eval --embedder local"
else
  echo "FAILED — see above."
fi
exit $(( fail > 0 ? 1 : 0 ))
