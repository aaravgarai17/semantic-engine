"""Golden evaluation set.

A small synthetic corpus with hand-labelled relevant chunks per question. It is
deliberately constructed so each retrieval method has questions it wins and
questions it loses:

  * **paraphrase** — the question shares no vocabulary with the answer. Dense
    retrieval should win; BM25 should fail outright.
  * **identifier** — the question is an exact code, error number, or API name.
    BM25 should win; dense should smear it into general semantic space.
  * **mixed** — both signals present, so fusion should be at least as good as
    the better of the two.

Building the set this way is the point. A corpus where dense always wins proves
nothing about hybrid retrieval; the interesting claim is that fusion covers
each method's blind spot, and you can only demonstrate that with questions that
land in those blind spots.

Everything is synthetic so the evaluation is reproducible with no external
data, no licensing questions, and no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QueryKind = Literal["paraphrase", "identifier", "mixed"]


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Question:
    query: str
    relevant_chunk_ids: list[str]
    kind: QueryKind
    note: str = ""


# --------------------------------------------------------------------- corpus
#
# Written so that vocabulary deliberately does *not* overlap between a question
# and its answer in the paraphrase cases.

DOCUMENTS: list[Document] = [
    Document(
        doc_id="auth",
        title="Authentication Guide",
        text="""# Authentication Guide

## Credential Recovery

If a member of staff can no longer sign in, initiate the credential recovery
workflow. The system dispatches a single-use link to the address on file. That
link stays valid for thirty minutes before it lapses.

## Session Lifetime

Sessions persist for eight hours of inactivity before terminating. Long-lived
sessions can be enabled per tenant but are discouraged for privileged accounts.

## Multi-Factor Enrolment

Second-factor enrolment uses time-based one-time codes. Hardware keys following
the WebAuthn specification are also accepted for accounts with elevated
privileges.
""",
    ),
    Document(
        doc_id="errors",
        title="Error Reference",
        text="""# Error Reference

## ERR_4032

ERR_4032 is emitted when the connection pool is exhausted and no handle becomes
free within the wait threshold. Raise the pool ceiling or reduce concurrency.

## ERR_5011

ERR_5011 indicates the upstream identity provider rejected the assertion. This
is almost always a clock skew problem between the two hosts.

## ERR_2004

ERR_2004 means the payload exceeded the size limit configured for the endpoint.
The default ceiling is four megabytes.
""",
    ),
    Document(
        doc_id="billing",
        title="Billing Operations",
        text="""# Billing Operations

## Invoice Generation

Statements are produced on the first day of each month covering the preceding
period. Generation runs in a background worker and typically finishes within
twenty minutes.

## Refund Handling

Reimbursements are processed back to the original instrument. Funds usually
appear within five to ten working days depending on the issuing bank.

## Proration

When a subscription tier changes mid-cycle, the charge is apportioned by the
number of days spent on each tier.
""",
    ),
    Document(
        doc_id="api",
        title="API Reference",
        text="""# API Reference

## POST /v2/transactions

The POST /v2/transactions endpoint records a new movement of funds. It requires
an idempotency_key header to make retries safe.

## GET /v2/accounts/{id}/balance

Returns the current cleared and pending amounts for an account. Results are
cached for five seconds.

## Rate Limits

The default allowance is 1000 requests per minute per API key. Exceeding it
returns HTTP 429 with a Retry-After header.
""",
    ),
    Document(
        doc_id="deploy",
        title="Deployment Runbook",
        text="""# Deployment Runbook

## Rolling Releases

New versions are introduced gradually across the fleet, replacing a quarter of
instances at a time and pausing for health verification between waves.

## Reverting a Release

To back out a bad version, promote the previously tagged image and re-run the
pipeline. The whole reversal typically completes in under four minutes.

## Database Migrations

Schema changes are applied ahead of the code that depends on them, and every
migration must be backward compatible with the version currently running.
""",
    ),

    # ------------------------------------------------------------------
    # Distractors.
    #
    # Without these the corpus is ~15 chunks and recall@5 is trivial — a
    # random guess covers a third of the corpus, so every method scores high
    # and the comparison measures nothing.
    #
    # These are deliberately *adjacent* in topic rather than unrelated. A
    # document about unrelated subject matter is easy to rank down; documents
    # that discuss sessions, timeouts, limits, and error codes without
    # containing the answer are what actually make retrieval hard, and they
    # are what a real corpus looks like.
    # ------------------------------------------------------------------
    Document(
        doc_id="security",
        title="Security Policy",
        text="""# Security Policy

## Password Requirements

Chosen secrets must be at least twelve characters and are checked against a
list of known-breached values at the point of selection.

## Account Lockout

After ten consecutive unsuccessful attempts an account is suspended for fifteen
minutes. Administrators may clear the suspension early.

## Audit Logging

Every authentication event is written to the immutable audit stream, including
failures, and retained for two years.

## Encryption at Rest

Stored records are encrypted with keys rotated every ninety days. Key material
never leaves the hardware module.
""",
    ),
    Document(
        doc_id="onboarding",
        title="Customer Onboarding",
        text="""# Customer Onboarding

## Identity Checks

New organisations complete verification before transacting. Documentation is
reviewed within two business days of submission.

## Sandbox Access

A sandbox environment is provisioned immediately and carries no verification
requirement. Sandbox data is purged weekly.

## Going Live

Production credentials are issued once verification concludes. The first live
movement of funds is held for manual inspection.
""",
    ),
    Document(
        doc_id="webhooks",
        title="Webhook Delivery",
        text="""# Webhook Delivery

## Retry Behaviour

Failed deliveries are retried with exponential backoff for up to twenty-four
hours before being abandoned.

## Signature Verification

Every payload carries an HMAC header computed with the endpoint secret.
Recipients should reject any request whose signature does not match.

## Event Ordering

Delivery order is not guaranteed. Consumers must tolerate arrival out of
sequence and should key on the event identifier.

## Timeouts

An endpoint that does not respond within ten seconds is treated as having
failed, and the delivery is queued for retry.
""",
    ),
    Document(
        doc_id="performance",
        title="Performance Notes",
        text="""# Performance Notes

## Connection Pooling

Handles are reused rather than established per request. The default ceiling is
twenty per instance, tunable through configuration.

## Query Timeouts

Statements exceeding thirty seconds are cancelled and surface as a timeout to
the caller.

## Caching Layer

Frequently requested reads are held in memory for sixty seconds. Writes
invalidate the corresponding entries immediately.

## Batch Processing

Bulk operations are chunked into groups of five hundred to bound memory use.
""",
    ),
    Document(
        doc_id="support",
        title="Support Procedures",
        text="""# Support Procedures

## Response Targets

Urgent reports receive a first response within one hour. Routine questions are
answered within one business day.

## Escalation Path

Unresolved urgent reports move to the on-call engineer after two hours, and to
the engineering lead after four.

## Status Communication

Incidents affecting multiple customers are published to the status page within
fifteen minutes of confirmation.
""",
    ),
    Document(
        doc_id="data",
        title="Data Handling",
        text="""# Data Handling

## Retention Windows

Transaction records are held for seven years to satisfy regulatory obligations.
Diagnostic logs are discarded after thirty days.

## Export Requests

Account holders may request a complete copy of their records, produced as JSON
within seven days of the request.

## Deletion Requests

Erasure requests are honoured except where retention is legally mandated.
Records that must be kept are pseudonymised instead.

## Regional Storage

Records are stored in the region selected at account creation and are not
replicated across regional boundaries.
""",
    ),
    Document(
        doc_id="integrations",
        title="Integration Guide",
        text="""# Integration Guide

## SDK Availability

Client libraries are published for Python, JavaScript, Ruby, and Go. Each
tracks the same major version as the API.

## Idempotent Requests

Any mutating call accepts a client-supplied key so that a retried request does
not duplicate its effect.

## Pagination

Collection endpoints return a cursor rather than an offset. Cursors remain
valid for one hour.

## Versioning

Breaking changes ship under a new major version. Prior versions are supported
for eighteen months after a successor is released.
""",
    ),
    Document(
        doc_id="internals",
        title="Implementation Notes",
        text="""# Implementation Notes

## Build Dependencies

Compiling the client from source on Debian requires libpq-dev to be present
before the build begins, along with the standard toolchain.

## Queue Primitives

Atomic job claiming uses BLMOVE to shift an entry from the pending list into a
per-worker processing list in a single operation. A separate pop followed by a
push would leave a window in which a crash loses the job.

## Vector Index Configuration

The similarity index is declared with vector_cosine_ops rather than the L2
operator class, because stored embeddings are unit normalised and cosine is the
meaningful comparison for them.

## Full Text Ranking

Keyword relevance is computed with ts_rank_cd, which accounts for the proximity
of matched terms rather than treating a document as an unordered bag.
""",
    ),
    Document(
        doc_id="monitoring",
        title="Monitoring and Alerting",
        text="""# Monitoring and Alerting

## Health Endpoints

Each service exposes a liveness path returning a status document. Orchestrators
poll it every ten seconds.

## Metric Collection

Counters and histograms are scraped at five-second intervals and retained at
full resolution for a fortnight.

## Alert Thresholds

A page is raised when the error ratio exceeds one percent over five minutes, or
when latency at the ninety-ninth percentile passes two seconds.

## On-Call Rotation

The rotation changes weekly at Monday midday. Handover notes are mandatory.
""",
    ),
]


# ------------------------------------------------------------------ questions
#
# `relevant_chunk_ids` are filled in at load time by matching a distinctive
# phrase, so the labels survive a change to chunk sizes.

QUESTIONS: list[tuple[str, str, QueryKind, str]] = [
    # ---- paraphrase: no vocabulary overlap with the answer ----
    (
        "how do I reset my password",
        "single-use link",
        "paraphrase",
        "answer says 'credential recovery', question says 'reset password'",
    ),
    (
        "how long until I get logged out automatically",
        "eight hours of inactivity",
        "paraphrase",
        "answer says 'sessions persist', question says 'logged out'",
    ),
    (
        "can I use a security key to log in",
        "WebAuthn specification",
        "paraphrase",
        "answer says 'hardware keys', question says 'security key'",
    ),
    (
        "when do customers get their money back",
        "original instrument",
        "paraphrase",
        "answer says 'reimbursements', question says 'money back'",
    ),
    (
        "how do I undo a bad deployment",
        "previously tagged image",
        "paraphrase",
        "answer says 'reverting a release', question says 'undo a bad deployment'",
    ),
    (
        "what happens if I upgrade halfway through the month",
        "apportioned by the number of days",
        "paraphrase",
        "answer says 'proration', question avoids the term",
    ),
    (
        "how often are bills sent out",
        "first day of each month",
        "paraphrase",
        "answer says 'statements are produced', question says 'bills sent'",
    ),
    (
        "is there a cap on how many calls I can make",
        "1000 requests per minute",
        "paraphrase",
        "answer says 'allowance', question says 'cap'",
    ),

    # ---- identifier: exact strings embeddings tend to smear ----
    (
        "ERR_4032",
        "connection pool is exhausted",
        "identifier",
        "bare error code, no semantic content at all",
    ),
    (
        "ERR_5011",
        "identity provider rejected",
        "identifier",
        "bare error code",
    ),
    (
        "ERR_2004",
        "payload exceeded the size limit",
        "identifier",
        "bare error code",
    ),
    (
        "POST /v2/transactions",
        "idempotency_key",
        "identifier",
        "exact endpoint path",
    ),
    (
        "GET /v2/accounts/{id}/balance",
        "cleared and pending",
        "identifier",
        "exact endpoint path with a parameter",
    ),
    (
        "WebAuthn",
        "WebAuthn specification",
        "identifier",
        "exact proper noun",
    ),
    (
        "idempotency_key header",
        "idempotency_key",
        "identifier",
        "exact parameter name",
    ),
    (
        "HTTP 429 Retry-After",
        "Retry-After header",
        "identifier",
        "exact status code and header name",
    ),

    # ---- mixed: both signals available ----
    (
        "what does error ERR_4032 mean and how do I fix it",
        "connection pool is exhausted",
        "mixed",
        "identifier plus natural language",
    ),
    (
        "database migration rules for deployments",
        "backward compatible",
        "mixed",
        "shared vocabulary and semantic intent",
    ),
    (
        "how are rolling releases performed across the fleet",
        "quarter of instances at a time",
        "mixed",
        "heavy vocabulary overlap",
    ),
    (
        "multi-factor enrolment with one-time codes",
        "time-based one-time codes",
        "mixed",
        "near-exact phrase match",
    ),
    (
        "what is the rate limit for the API",
        "1000 requests per minute",
        "mixed",
        "vocabulary overlap plus intent",
    ),
    (
        "how long does invoice generation take",
        "twenty minutes",
        "mixed",
        "shared term 'invoice generation'",
    ),

    # ------------------------------------------------------------------
    # Second batch, taking the set from 22 to 40 questions.
    #
    # Weighted toward the cases most likely to separate the retrievers:
    # harder paraphrases (where the answer's vocabulary is further from the
    # question's) and identifiers unlikely to appear in an embedding model's
    # training data.
    # ------------------------------------------------------------------

    # ---- paraphrase ----
    (
        "what stops someone guessing my login over and over",
        "ten consecutive unsuccessful attempts",
        "paraphrase",
        "answer says 'account lockout', question describes brute forcing",
    ),
    (
        "is my information scrambled when it is sitting on disk",
        "encrypted with keys rotated",
        "paraphrase",
        "answer says 'encryption at rest', question paraphrases heavily",
    ),
    (
        "how quickly will someone get back to me if everything is broken",
        "first response within one hour",
        "paraphrase",
        "answer says 'urgent reports', question describes the situation",
    ),
    (
        "can I get everything you hold about me",
        "complete copy of their records",
        "paraphrase",
        "answer says 'export requests', question paraphrases",
    ),
    (
        "what happens to my records if I close my account",
        "erasure requests are honoured",
        "paraphrase",
        "answer says 'deletion requests', question describes the scenario",
    ),
    (
        "how do you stop the same charge going through twice",
        "client-supplied key",
        "paraphrase",
        "answer says 'idempotent requests', question describes the effect",
    ),
    (
        "who gets woken up when something breaks at night",
        "rotation changes weekly",
        "paraphrase",
        "answer says 'on-call rotation', question is colloquial",
    ),
    (
        "what if my server is slow to accept a notification",
        "does not respond within ten seconds",
        "paraphrase",
        "answer says 'timeouts', question describes the symptom",
    ),
    (
        "how do you stop one enormous request eating all the memory",
        "chunked into groups of five hundred",
        "paraphrase",
        "answer says 'batch processing', question describes the concern",
    ),
    (
        "do I have to prove who I am before I can move real money",
        "verification before transacting",
        "paraphrase",
        "answer says 'identity checks', question is conversational",
    ),

    # ---- identifier ----
    (
        "libpq-dev",
        "libpq-dev",
        "identifier",
        "exact package name",
    ),
    (
        "BLMOVE",
        "BLMOVE",
        "identifier",
        "exact command name (only appears once in the corpus)",
    ),
    (
        "vector_cosine_ops",
        "vector_cosine_ops",
        "identifier",
        "exact index operator class",
    ),
    (
        "ts_rank_cd",
        "ts_rank_cd",
        "identifier",
        "exact function name",
    ),

    # ---- mixed ----
    (
        "what is the retention period for transaction records",
        "seven years",
        "mixed",
        "shared vocabulary plus a specific fact",
    ),
    (
        "how are webhook failures retried",
        "exponential backoff",
        "mixed",
        "shared term 'retried' plus intent",
    ),
    (
        "which SDK languages are supported",
        "Python, JavaScript, Ruby, and Go",
        "mixed",
        "shared term 'SDK'",
    ),
    (
        "when does an alert page someone",
        "error ratio exceeds one percent",
        "mixed",
        "shared term 'alert' plus semantic intent",
    ),
]


def _flatten(text: str) -> str:
    """Collapse all whitespace so matching survives line wrapping.

    Source documents wrap at 80 columns, so a golden phrase like "apportioned
    by the number of days" is split across a newline in the chunk text and an
    exact substring match fails. Normalising both sides makes the labels depend
    on the words rather than on where the author happened to break the line.
    """
    return " ".join(text.lower().split())


def build_questions(chunks_by_id: dict[str, str]) -> list[Question]:
    """Resolve each question's answer phrase to the chunk ids containing it.

    Labelling by phrase rather than by hard-coded chunk id means the golden set
    survives a change to chunk size or overlap — otherwise every chunking
    experiment would silently invalidate the labels, and the experiment would
    measure nothing.
    """
    questions: list[Question] = []

    for query, phrase, kind, note in QUESTIONS:
        needle = _flatten(phrase)
        matching = [
            cid for cid, text in chunks_by_id.items() if needle in _flatten(text)
        ]
        if not matching:
            raise ValueError(
                f"golden answer phrase {phrase!r} for query {query!r} matched no "
                "chunk — the corpus and the question set have drifted apart"
            )
        questions.append(
            Question(query=query, relevant_chunk_ids=matching, kind=kind, note=note)
        )

    return questions
