#!/usr/bin/env python3
"""
HEP Research Library - Search Application
A Flask web app for querying the HEP research PDF library.
Uses Claude API for answer generation and hybrid Qdrant search.
"""

import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
from urllib.parse import quote
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, abort, redirect, Response
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import anthropic
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = Path(os.environ.get("HEP_LOGO_PATH", str(BASE_DIR / "HEP logo white.png")))


def _clean_env_str(val: str | None) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


VOYAGE_API_KEY       = _clean_env_str(os.environ.get("VOYAGE_API_KEY"))
VOYAGE_MODEL         = os.environ.get("VOYAGE_MODEL", "voyage-3").strip() or "voyage-3"
VOYAGE_RERANK_MODEL  = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2").strip() or "rerank-2"
RERANK_ENABLED       = os.environ.get("RERANK_ENABLED", "true").strip().lower() not in ("0", "false", "no")
HYBRID_TOP_K         = int(os.environ.get("HYBRID_TOP_K", "20"))
FINAL_TOP_K          = int(os.environ.get("FINAL_TOP_K",  "8"))

# Phase 3 flags
CONTRADICTION_ENABLED = os.environ.get("CONTRADICTION_ENABLED", "true").strip().lower() not in ("0", "false", "no")

# Model configuration
ANSWER_MODEL        = os.environ.get("ANSWER_MODEL", "claude-opus-4-5").strip() or "claude-opus-4-5"
CONTRADICTION_MODEL = os.environ.get("CONTRADICTION_MODEL", "claude-sonnet-4-5-20250514").strip() or "claude-sonnet-4-5-20250514"

QDRANT_URL = _clean_env_str(os.environ.get("QDRANT_URL"))


def _read_qdrant_api_key() -> str:
    key = (
        _clean_env_str(os.environ.get("QDRANT_API_KEY"))
        or _clean_env_str(os.environ.get("HEP_QDRANT_API_KEY"))
        or _clean_env_str(os.environ.get("QDRANT_KEY"))
        or ""
    )
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


QDRANT_API_KEY    = _read_qdrant_api_key()
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "hep_research").strip() or "hep_research"

HEP_PDF_BASE_URL = os.environ.get(
    "HEP_PDF_BASE_URL",
    "https://pub-40a226eaa06b4f59988b1d1213d57d9c.r2.dev/pdfs",
).strip().rstrip("/")

client = anthropic.Anthropic()
app    = Flask(__name__)

_max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "100"))
app.config["MAX_CONTENT_LENGTH"] = _max_upload_mb * 1024 * 1024

try:
    from hep_export import export_pdf, export_docx
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    export_pdf = export_docx = None

try:
    from pdf_ingest import ingest_pdf, upload_config_detail, upload_config_ok
except ImportError:
    ingest_pdf = None

    def upload_config_detail() -> dict:
        return {
            "ok": False,
            "checks": [{
                "id": "pdf_ingest_import",
                "name": "pdf_ingest module",
                "ok": False,
                "detail": "not importable",
            }],
        }

    def upload_config_ok() -> bool:
        return False

# ── HTTP Basic Auth ────────────────────────────────────────────────────────────

AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "").strip()
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "").strip()
AUTH_ENABLED  = bool(AUTH_USERNAME and AUTH_PASSWORD)


def _unauthorized() -> Response:
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="HEP Research Library"'},
    )


@app.errorhandler(RequestEntityTooLarge)
def _payload_too_large(_e):
    return jsonify({"ok": False, "error": "File too large", "steps": []}), 413


@app.before_request
def _require_basic_auth():
    if not AUTH_ENABLED:
        return None
    # These routes must remain open (health check, static assets)
    if request.path in ("/health", "/logo"):
        return None
    # Admin is open — accessible after logging into the main site
    if request.path.startswith("/admin"):
        return None
    # Everything else requires authentication
    auth = request.authorization
    if not auth or auth.username != AUTH_USERNAME or auth.password != AUTH_PASSWORD:
        return _unauthorized()
    return None


_voy    = None
_qdrant = None

# BM25 corpus cache — avoids re-scrolling Qdrant on every search
_bm25_cache: dict[str, list] = {}   # category -> records list
_bm25_cache_ts: dict[str, float] = {}  # category -> timestamp
_BM25_CACHE_TTL = 300  # seconds (5 minutes)

# Document list cache — avoids re-scrolling on every /documents page load
_docs_cache: list | None = None
_docs_cache_ts: float = 0
_DOCS_CACHE_TTL = 300  # seconds (5 minutes)


def get_voyage_client() -> voyageai.Client:
    global _voy
    if _voy is None:
        if not VOYAGE_API_KEY:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        _voy = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voy


def get_qdrant_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set")
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return _qdrant


def embed_query(text: str) -> list:
    voy  = get_voyage_client()
    resp = voy.embed(texts=[text], model=VOYAGE_MODEL)
    return resp.embeddings[0]


def pdf_url_for_filename(filename: str) -> str:
    if not filename:
        return ""
    return f"{HEP_PDF_BASE_URL}/{quote(filename, safe='')}"


# ── Keyword / BM25 Helpers ────────────────────────────────────────────────────

def _tokenise_query(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def _scroll_by_category(qdrant: QdrantClient, category: str, limit: int = 2000) -> list:
    filt = None
    if category and category != "All":
        filt = qm.Filter(
            must=[qm.FieldCondition(key="category", match=qm.MatchValue(value=category))]
        )
    records, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=filt,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return records


def _cached_scroll(qdrant: QdrantClient, category: str) -> list:
    """Return BM25 corpus from cache or fresh scroll. TTL = 5 minutes."""
    now = time.monotonic()
    key = category or "All"
    if key in _bm25_cache and (now - _bm25_cache_ts.get(key, 0)) < _BM25_CACHE_TTL:
        return _bm25_cache[key]
    records = _scroll_by_category(qdrant, category, limit=2000)
    _bm25_cache[key] = records
    _bm25_cache_ts[key] = now
    return records


def _invalidate_bm25_cache():
    """Clear BM25 cache after corpus changes (e.g. PDF upload)."""
    _bm25_cache.clear()
    _bm25_cache_ts.clear()


def _invalidate_docs_cache():
    """Clear document list cache after corpus changes."""
    global _docs_cache, _docs_cache_ts
    _docs_cache = None
    _docs_cache_ts = 0


def _get_all_docs_cached() -> list:
    """Return full deduplicated document list from cache or fresh scroll."""
    global _docs_cache, _docs_cache_ts
    now = time.monotonic()
    if _docs_cache is not None and (now - _docs_cache_ts) < _DOCS_CACHE_TTL:
        return _docs_cache

    qdrant = get_qdrant_client()
    records, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=5000,
        with_payload=True,
        with_vectors=False,
    )
    seen = {}
    for r in records:
        p  = dict(r.payload or {})
        fn = p.get("filename", "")
        if fn and fn not in seen:
            seen[fn] = p
    docs = list(seen.values())
    docs.sort(key=lambda d: (d.get("title") or d.get("filename") or "").lower())
    _docs_cache = docs
    _docs_cache_ts = now
    return _docs_cache


def _bm25_search(query: str, records: list, top_k: int) -> list[tuple[float, dict]]:
    from math import log
    tokens = _tokenise_query(query)
    if not tokens or not records:
        return []

    corpus: list[list[str]] = []
    for r in records:
        p    = dict(r.payload or {})
        text = " ".join(filter(None, [p.get("title", ""), p.get("text", ""), p.get("authors", "")]))
        corpus.append(_tokenise_query(text))

    N     = len(corpus)
    k1, b = 1.5, 0.75
    avgdl = sum(len(d) for d in corpus) / N if N else 1

    idf: dict[str, float] = {}
    for t in set(tokens):
        df      = sum(1 for d in corpus if t in d)
        idf[t]  = log((N - df + 0.5) / (df + 0.5) + 1)

    scores: list[float] = []
    for doc in corpus:
        dl   = len(doc)
        freq = {}
        for w in doc:
            freq[w] = freq.get(w, 0) + 1
        s = sum(
            idf.get(t, 0) * (freq.get(t, 0) * (k1 + 1))
            / (freq.get(t, 0) + k1 * (1 - b + b * dl / avgdl))
            for t in tokens
        )
        scores.append(s)

    ranked = sorted(zip(scores, records), key=lambda x: x[0], reverse=True)[:top_k]
    return [(sc, dict(r.payload or {})) for sc, r in ranked if sc > 0]


def _rrf_merge(
    vector_hits:  list[tuple[float, dict]],
    keyword_hits: list[tuple[float, dict]],
    k: int = 60,
    top_n: int = 20,
) -> list[tuple[float, dict]]:
    def _key(p: dict) -> tuple:
        return (p.get("filename", ""), p.get("chunk_index", -1))

    rrf: dict[tuple, float] = {}
    pays: dict[tuple, dict] = {}

    for rank, (_, p) in enumerate(vector_hits, start=1):
        key       = _key(p)
        rrf[key]  = rrf.get(key, 0.0) + 1.0 / (k + rank)
        pays[key] = p

    for rank, (_, p) in enumerate(keyword_hits, start=1):
        key       = _key(p)
        rrf[key]  = rrf.get(key, 0.0) + 1.0 / (k + rank)
        pays[key] = p

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(score, pays[key]) for key, score in ranked]


def _rerank(query: str, candidates: list[tuple[float, dict]], top_n: int) -> list[tuple[float, dict]]:
    if not RERANK_ENABLED or not candidates:
        return candidates[:top_n]
    try:
        voy  = get_voyage_client()
        docs = [
            " ".join(filter(None, [p.get("title", ""), p.get("text", "")[:1000]]))
            for _, p in candidates
        ]
        result   = voy.rerank(query=query, documents=docs, model=VOYAGE_RERANK_MODEL, top_k=top_n)
        reranked = [(float(item.relevance_score), candidates[item.index][1]) for item in result.results]
        return reranked
    except Exception as e:
        print(f"[rerank] warning: {e} — falling back to RRF order")
        return candidates[:top_n]


def semantic_search(query: str, category: str = "All", top_k: int = 8) -> list:
    """Hybrid retrieval: vector + BM25 via RRF, then Voyage rerank."""
    qdrant = get_qdrant_client()

    vec  = embed_query(query)
    filt = None
    if category and category != "All":
        filt = qm.Filter(
            must=[qm.FieldCondition(key="category", match=qm.MatchValue(value=category))]
        )
    response     = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec, limit=HYBRID_TOP_K,
        with_payload=True, with_vectors=False,
        query_filter=filt,
    )
    vector_hits  = [(float(h.score), dict(h.payload or {})) for h in response.points]
    records      = _cached_scroll(qdrant, category)
    keyword_hits = _bm25_search(query, records, top_k=HYBRID_TOP_K)
    merged       = _rrf_merge(vector_hits, keyword_hits, top_n=HYBRID_TOP_K)
    return _rerank(query, merged, top_n=FINAL_TOP_K)


# ── Phase 3: Evidence Strength ────────────────────────────────────────────────

def compute_evidence_strength(citations: list[dict], answer_text: str) -> dict[int, int]:
    """
    For each citation index, count how many *independent documents* (distinct
    filenames) are cited alongside it in the same sentence of the answer.

    Returns {citation_index: independent_source_count}.
    The count represents how many distinct documents support the same
    part of the answer as this citation - including itself (minimum 1).
    """
    # Find all [n] references per sentence
    sentences = re.split(r'(?<=[.!?])\s+', answer_text)

    # Map filename -> set of citation indices pointing to that file
    filename_to_indices: dict[str, set[int]] = {}
    for c in citations:
        fn = c.get("filename", "")
        if fn:
            filename_to_indices.setdefault(fn, set()).add(c["index"])

    # For each citation, count distinct filenames cited in same sentences
    cooccurrence: dict[int, set[str]] = {c["index"]: set() for c in citations}
    for sentence in sentences:
        cited_nums = [int(m) for m in re.findall(r'\[(\d+)\]', sentence)]
        if not cited_nums:
            continue
        # Collect filenames for all citations in this sentence
        filenames_in_sentence = set()
        for n in cited_nums:
            matching = [c for c in citations if c["index"] == n]
            if matching:
                fn = matching[0].get("filename", "")
                if fn:
                    filenames_in_sentence.add(fn)
        # Credit each cited index with all filenames in sentence
        for n in cited_nums:
            if n in cooccurrence:
                cooccurrence[n] |= filenames_in_sentence

    # Also ensure each citation counts at least its own document
    for c in citations:
        fn = c.get("filename", "")
        if fn:
            cooccurrence[c["index"]].add(fn)

    return {idx: max(1, len(fns)) for idx, fns in cooccurrence.items()}


# ── Phase 3: Contradiction Detection ─────────────────────────────────────────

def detect_contradictions(query: str, results: list[tuple[float, dict]]) -> list[dict]:
    """
    Asks Claude to identify factual contradictions across the retrieved chunks.
    Returns a list of contradiction objects:
      [{ "topic": str, "sides": [{ "index": int, "claim": str }, ...] }]
    Returns [] if none found or if disabled.
    """
    if not CONTRADICTION_ENABLED or len(results) < 2:
        return []

    # Build a compact source list for the contradiction prompt
    source_lines = []
    for i, (_, doc) in enumerate(results, 1):
        title      = doc.get("title") or doc.get("filename") or "Unknown"
        year       = doc.get("year") or "n.d."
        chunk_text = (doc.get("text") or "")[:600]
        source_lines.append(f"[{i}] {title} ({year})\n{chunk_text}")

    sources_block = "\n\n".join(source_lines)

    prompt = f"""You are a research analyst checking for factual contradictions across research sources.

Research question: {query}

Below are {len(results)} source excerpts. Identify any direct factual contradictions - where two or more sources make conflicting specific claims about the same thing (statistics, dates, named findings, causal claims).

Do not flag differences in emphasis, methodology, or scope. Only flag direct factual conflicts.

If you find contradictions, respond in this exact JSON format:
{{
  "contradictions": [
    {{
      "topic": "brief description of what they disagree on",
      "sides": [
        {{"index": 1, "claim": "what source 1 says"}},
        {{"index": 3, "claim": "what source 3 says"}}
      ]
    }}
  ]
}}

If there are no direct factual contradictions, respond with:
{{"contradictions": []}}

Sources:
{sources_block}

JSON response only:"""

    try:
        message = client.messages.create(
            model=CONTRADICTION_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        import json
        parsed = json.loads(raw)
        return parsed.get("contradictions", [])
    except Exception as e:
        print(f"[contradiction] warning: {e}")
        return []


# ── Answer Generation ─────────────────────────────────────────────────────────

def _build_answer_context(query: str, results: list) -> tuple:
    """Build citations list and Claude prompt from search results.
    Returns (citations, prompt). Shared by generate_answer and search-stream."""
    if not results:
        return [], ""

    context_parts = []
    citations     = []
    doc_url_template = os.environ.get("HEP_DOC_URL_TEMPLATE", "").strip()

    for i, (score, doc) in enumerate(results, 1):
        title       = doc.get("title") or doc.get("filename") or "Unknown title"
        authors     = doc.get("authors") or "Unknown"
        year        = doc.get("year") or "n.d."
        filename    = doc.get("filename") or ""
        category    = doc.get("category") or ""
        chunk_index = doc.get("chunk_index")
        page_start  = doc.get("page_start")
        page_end    = doc.get("page_end")
        chunk_text  = doc.get("text") or ""

        ref_bits = []
        if page_start and page_end:
            ref_bits.append(f"pp. {page_start}-{page_end}")
        if chunk_index is not None:
            ref_bits.append(f"chunk {chunk_index}")
        ref = ("; " + ", ".join(ref_bits)) if ref_bits else ""

        context_parts.append(f"[{i}] {title} ({authors}, {year}){ref}\n{chunk_text[:2000]}")

        citation = {
            "index":           i,
            "title":           title,
            "authors":         authors,
            "year":            year,
            "filename":        filename,
            "category":        category,
            "relevance_score": round(score, 3),
            "page_start":      page_start,
            "page_end":        page_end,
            "chunk_index":     chunk_index,
            "pdf_url":         pdf_url_for_filename(filename),
            "excerpt":         chunk_text[:300] if chunk_text else "",
        }
        if doc_url_template and "{filename}" in doc_url_template:
            citation["url"] = doc_url_template.replace("{filename}", filename)
        citations.append(citation)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a research analyst for Hope Education Project, a survivor-led anti-trafficking NGO operating in Ghana and West Africa. You have deep field knowledge of trafficking patterns, enforcement systems, and survivor experience in the region.

Answer the following research question using ONLY the provided sources. Apply this analytical framework:

WHAT COUNTS AS STRONG EVIDENCE - lead with these:
- Documented prosecutions, convictions, and named perpetrators
- Specific statistics with named sources and dates
- Operational outcomes with verified numbers
- Legislative or policy changes with enforcement teeth
- Push-factor interventions addressing root vulnerability

TREAT WITH SCEPTICISM - flag these explicitly:
- Mass sensitisation campaigns where reach is mistaken for impact
- Interventions that show survivor numbers without survivor outcomes
- Claims without a named source, date, or methodology

ALWAYS:
- Distinguish trafficking from smuggling precisely (trafficking = exploitation is the purpose, crime continues at destination; smuggling = facilitated border crossing, crime ends at destination)
- Name specific geographies - Ghana and Nigeria are primary focus, then wider West Africa
- Name specific enforcement agencies where relevant (NAPTIP in Nigeria, EOCO in Ghana)
- Note when evidence is from a single source only
- Note gaps - what the research does not address
- Use plain declarative sentences. No hedging. No throat-clearing.
- Maximum 4 paragraphs
- Cite sources using [number] notation inline

Research question: {query}

Sources:
{context}

Answer:"""

    return citations, prompt


def generate_answer(query: str, results: list) -> dict:
    if not results:
        return {"answer": "No relevant documents found.", "citations": [], "contradictions": []}

    citations, prompt = _build_answer_context(query, results)

    # Start contradiction detection in parallel with answer generation
    with ThreadPoolExecutor(max_workers=1) as executor:
        contradiction_future = executor.submit(detect_contradictions, query, results)

        message = client.messages.create(
            model=ANSWER_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        answer_text = message.content[0].text.strip()

        # Evidence strength runs instantly (no API call)
        evidence_strength = compute_evidence_strength(citations, answer_text)
        for c in citations:
            c["evidence_strength"] = evidence_strength.get(c["index"], 1)

        # Collect contradiction result (already running or finished)
        contradictions = contradiction_future.result(timeout=30)

    return {
        "answer":        answer_text,
        "citations":     citations,
        "contradictions": contradictions,
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────


@app.route("/")
def home():
    doc_count = "Qdrant"
    logo_url  = os.environ.get("HEP_LOGO_URL", "").strip() or "/logo"
    return render_template("index.html", doc_count=doc_count, logo_url=logo_url)


@app.route("/logo")
def serve_logo():
    logo_url = os.environ.get("HEP_LOGO_URL", "").strip()
    if logo_url:
        return redirect(logo_url, code=302)
    if not LOGO_PATH.exists():
        abort(404)
    return send_file(str(LOGO_PATH), mimetype="image/png")


@app.route("/search", methods=["POST"])
def search_route():
    data     = request.get_json()
    query    = data.get("query", "").strip()
    category = data.get("category", "All")
    if not query:
        return jsonify({"error": "No query provided"})
    try:
        hits = semantic_search(query, category=category)
        return jsonify(generate_answer(query, hits))
    except Exception as e:
        return jsonify({"error": str(e)})


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/search-stream", methods=["POST"])
def search_stream_route():
    data     = request.get_json()
    query    = data.get("query", "").strip()
    category = data.get("category", "All")
    if not query:
        def _err():
            yield _sse_event("error", {"error": "No query provided"})
            yield _sse_event("done", {})
        return Response(_err(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def generate():
        try:
            hits = semantic_search(query, category=category)
            if not hits:
                yield _sse_event("error", {"error": "No relevant documents found."})
                yield _sse_event("done", {})
                return

            citations, prompt = _build_answer_context(query, hits)

            # Send citations immediately — before Claude starts
            yield _sse_event("citations", {"citations": citations})

            # Start contradiction detection in background
            with ThreadPoolExecutor(max_workers=1) as executor:
                contradiction_future = executor.submit(detect_contradictions, query, hits)

                # Stream answer from Claude
                full_answer = []
                with client.messages.stream(
                    model=ANSWER_MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        full_answer.append(text)
                        yield _sse_event("answer_chunk", {"text": text})

                answer_text = "".join(full_answer).strip()
                yield _sse_event("answer_done", {})

                # Evidence strength (instant — no API call)
                evidence = compute_evidence_strength(citations, answer_text)
                yield _sse_event("evidence", {"evidence": {str(k): v for k, v in evidence.items()}})

                # Collect contradiction result
                try:
                    contradictions = contradiction_future.result(timeout=30)
                except Exception:
                    contradictions = []
                yield _sse_event("contradictions", {"contradictions": contradictions})

        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

        yield _sse_event("done", {})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/similar", methods=["POST"])
def similar_route():
    """
    Find documents similar to a given document using its own vector.
    Fetches the document's embedding from Qdrant by filename,
    then queries for nearest neighbours. Returns deduplicated by filename.
    """
    data     = request.get_json()
    filename = data.get("filename", "").strip()
    title    = data.get("title", filename)
    limit    = int(data.get("limit", 12))

    if not filename:
        return jsonify({"ok": False, "error": "No filename provided"}), 400

    try:
        qdrant = get_qdrant_client()

        # Step 1: find a chunk from this document to get its vector
        scroll_result, _ = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(
                    key="filename",
                    match=qm.MatchValue(value=filename),
                )]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,   # need the vector
        )

        if not scroll_result:
            return jsonify({"ok": False, "error": f"Document not found in index: {filename}"}), 404

        source_vector = scroll_result[0].vector

        # Step 2: query for nearest neighbours, excluding the source document
        response = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=source_vector,
            limit=limit * 4,    # fetch extra to allow dedup and exclusion
            with_payload=True,
            with_vectors=False,
        )

        # Step 3: deduplicate by filename, exclude the source document
        seen = {}
        for hit in response.points:
            p  = dict(hit.payload or {})
            fn = p.get("filename", "")
            if fn and fn != filename and fn not in seen:
                seen[fn] = (float(hit.score), p)

        # Sort by score, take top limit
        ranked = sorted(seen.values(), key=lambda x: x[0], reverse=True)[:limit]

        results = []
        for score, doc in ranked:
            results.append({
                "title":    doc.get("title") or doc.get("filename") or "Unknown",
                "authors":  doc.get("authors") or "",
                "year":     doc.get("year") or "",
                "category": doc.get("category") or "",
                "filename": doc.get("filename") or "",
                "excerpt":  (doc.get("text") or "")[:200],
                "pdf_url":  pdf_url_for_filename(doc.get("filename") or ""),
                "score":    round(score, 3),
            })

        return jsonify({"ok": True, "source_title": title, "documents": results})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/documents", methods=["GET"])
def documents_route():
    """
    Full corpus browser - returns all documents deduplicated by filename.
    Supports optional ?category= and ?q= (title search) query params.
    Uses server-side cache refreshed every 5 minutes or on upload.
    """
    category = request.args.get("category", "All").strip()
    q        = request.args.get("q", "").strip().lower()

    try:
        docs = _get_all_docs_cached()

        # Filter by category if specified
        if category and category != "All":
            docs = [d for d in docs if d.get("category") == category]

        # Filter by title/author search if provided
        if q:
            docs = [d for d in docs if q in (d.get("title") or d.get("filename") or "").lower()
                    or q in (d.get("authors") or "").lower()]

        results = [{
            "title":    d.get("title") or d.get("filename") or "Unknown",
            "authors":  d.get("authors") or "",
            "year":     d.get("year") or "",
            "category": d.get("category") or "",
            "filename": d.get("filename") or "",
            "pdf_url":  pdf_url_for_filename(d.get("filename") or ""),
        } for d in docs]

        return jsonify({"ok": True, "total": len(results), "documents": results})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/discover", methods=["POST"])
def discover_route():
    """
    Discovery mode: return a sample of documents for a category
    without a search query. Uses Qdrant scroll, deduplicates by filename,
    returns one representative chunk per document.
    """
    data     = request.get_json()
    category = data.get("category", "All")
    limit    = int(data.get("limit", 12))

    try:
        qdrant = get_qdrant_client()
        filt   = None
        if category and category != "All":
            filt = qm.Filter(
                must=[qm.FieldCondition(key="category", match=qm.MatchValue(value=category))]
            )
        # Paginate to get full category corpus
        all_records = []
        offset = None
        while True:
            batch, offset = qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=filt,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_records.extend(batch)
            if offset is None:
                break
        records = all_records

        # One chunk per document (first chunk encountered)
        seen  = {}
        for r in records:
            p  = dict(r.payload or {})
            fn = p.get("filename", "")
            if fn and fn not in seen:
                seen[fn] = p

        # Sort by title, return up to limit
        docs = sorted(seen.values(), key=lambda x: (x.get("title") or x.get("filename") or "").lower())
        docs = docs[:limit]

        results = []
        for doc in docs:
            results.append({
                "title":    doc.get("title") or doc.get("filename") or "Unknown",
                "authors":  doc.get("authors") or "",
                "year":     doc.get("year") or "",
                "category": doc.get("category") or "",
                "filename": doc.get("filename") or "",
                "excerpt":  (doc.get("text") or "")[:200],
                "pdf_url":  pdf_url_for_filename(doc.get("filename") or ""),
            })

        return jsonify({"ok": True, "category": category, "documents": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/stats")
def stats_route():
    """Public-facing corpus statistics for the main UI."""
    try:
        qc      = get_qdrant_client()
        chunks  = qc.count(collection_name=QDRANT_COLLECTION, exact=True).count
        return jsonify({"ok": True, "chunks": chunks})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "chunks": None})


@app.route("/export", methods=["POST"])
def export_route():
    if not EXPORT_AVAILABLE:
        return jsonify({"error": "Export not available - hep_export module missing"}), 500

    data          = request.get_json()
    fmt           = data.get("format", "pdf").lower()   # "pdf" or "docx"
    query         = data.get("query", "Untitled query")
    answer        = data.get("answer", "")
    citations     = data.get("citations", [])
    contradictions = data.get("contradictions", [])

    try:
        if fmt == "docx":
            file_bytes = export_docx(query, answer, citations, contradictions)
            mime       = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename   = "HEP-Research-Export.docx"
        else:
            file_bytes = export_pdf(query, answer, citations, contradictions)
            mime       = "application/pdf"
            filename   = "HEP-Research-Export.pdf"

        buf = io.BytesIO(file_bytes)
        buf.seek(0)
        return send_file(
            buf,
            mimetype=mime,
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "ok":                   True,
        "qdrant_configured":    bool(QDRANT_URL and QDRANT_API_KEY),
        "hybrid_retrieval":     True,
        "rerank_enabled":       RERANK_ENABLED,
        "contradiction_enabled": CONTRADICTION_ENABLED,
        "answer_model":         ANSWER_MODEL,
        "contradiction_model":  CONTRADICTION_MODEL,
    }), 200


@app.route("/admin")
def admin_page():
    logo_url  = os.environ.get("HEP_LOGO_URL", "").strip() or "/logo"
    ingest_ok = ingest_pdf is not None
    return render_template(
        "admin.html",
        logo_url=logo_url,
        ingest_available=ingest_ok,
        upload_ready=bool(ingest_ok and upload_config_ok()),
        max_mb=_max_upload_mb,
    )


@app.route("/admin/api/status")
def admin_status():
    try:
        qc  = get_qdrant_client()
        cnt = qc.count(collection_name=QDRANT_COLLECTION, exact=True).count
        return jsonify({"ok": True, "total_chunks": cnt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "total_chunks": None}), 200


_ADMIN_DEBUG_ENV_KEYS = (
    "QDRANT_URL", "QDRANT_API_KEY", "HEP_QDRANT_API_KEY", "VOYAGE_API_KEY",
    "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ENDPOINT_URL",
    "GOOGLE_SERVICE_ACCOUNT_URL", "GOOGLE_SERVICE_ACCOUNT_JSON", "GOOGLE_DRIVE_FOLDER_ID",
)


@app.route("/admin/api/debug")
def admin_debug():
    return jsonify({k: bool(_clean_env_str(os.environ.get(k))) for k in _ADMIN_DEBUG_ENV_KEYS})


@app.route("/admin/api/upload-debug")
def admin_upload_debug():
    return jsonify(upload_config_detail())


@app.route("/admin/api/upload", methods=["POST"])
def admin_upload():
    if ingest_pdf is None:
        return jsonify({"ok": False, "error": "pdf_ingest module not available", "steps": []}), 500
    if not upload_config_ok():
        return jsonify({"ok": False, "error": "Missing R2 or Google Drive environment variables", "steps": []}), 400

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file provided", "steps": []}), 400
    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "Only PDF files are allowed", "steps": []}), 400

    tmp     = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = Path(tmp.name)
    try:
        tmp.close()
        f.save(tmp_path)
        result = ingest_pdf(tmp_path, filename)
        code   = 200 if result.get("ok") else 500
        if result.get("ok"):
            _invalidate_bm25_cache()
            _invalidate_docs_cache()
        return jsonify({
            "ok":              bool(result.get("ok")),
            "steps":           result.get("steps", []),
            "error":           result.get("error"),
            "chunks_upserted": result.get("chunks_upserted", 0),
        }), code
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HEP Research Library")
    print("=" * 50)
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
