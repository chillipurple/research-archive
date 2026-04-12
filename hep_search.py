#!/usr/bin/env python3
"""
HEP Research Library - Search Application
A Flask web app for querying the HEP research PDF library.
Uses Claude API for answer generation and TF-IDF for search.
"""

import io
import re
import csv
import os
import pickle
import tempfile
import urllib.request
import urllib.error
from urllib.parse import quote
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, abort, redirect, Response
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import anthropic
import fitz  # pymupdf
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = Path(os.environ.get("HEP_LOGO_PATH", str(BASE_DIR / "HEP logo white.png")))

RESEARCH_DIR_ENV = os.environ.get("HEP_RESEARCH_DIR")
RESEARCH_DIR = Path(RESEARCH_DIR_ENV) if RESEARCH_DIR_ENV else None
OUTPUT_CSV = (RESEARCH_DIR / "_research_index.csv") if RESEARCH_DIR else None

INDEX_URL = os.environ.get("HEP_INDEX_URL", "").strip() or None
INDEX_FILE = Path(os.environ.get("HEP_INDEX_FILE", str(BASE_DIR / "data" / "_vector_index.pkl"))).resolve()


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
    if request.path == "/health":
        return None
    auth = request.authorization
    if not auth or auth.username != AUTH_USERNAME or auth.password != AUTH_PASSWORD:
        return _unauthorized()
    return None

# ── Text and Vector Helpers ───────────────────────────────────────────────────

def extract_text(pdf_path: Path, max_chars: int = 8000) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page_num in range(min(8, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        return text[:max_chars].strip()
    except Exception:
        return ""


def tokenise(text: str) -> list:
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


def build_vocab(documents: list) -> dict:
    n_docs = len(documents)
    word_doc_counts = {}
    for doc in documents:
        words = set(tokenise(doc["combined"]))
        for word in words:
            word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
    filtered = sorted(
        word for word, count in word_doc_counts.items()
        if 2 <= count <= n_docs * 0.5
    )
    return {word: idx for idx, word in enumerate(filtered)}


def embed(text: str, vocab: dict) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for word in tokenise(text):
        idx = vocab.get(word)
        if idx is not None and idx < len(vec):
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


_voy    = None
_qdrant = None


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


# ── Index Building ────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HEPResearchLibrary/1.0; +https://hopeeducationproject.org)",
                "Accept": "*/*",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def ensure_index_present() -> None:
    if INDEX_FILE.exists():
        return
    if not INDEX_URL:
        raise FileNotFoundError(
            f"Index file not found at '{INDEX_FILE}'. "
            "Set HEP_INDEX_URL to a downloadable _vector_index.pkl for cloud deployment."
        )
    print(f"Downloading index from {INDEX_URL} ...")
    try:
        _download_file(INDEX_URL, INDEX_FILE)
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Failed to download index from HEP_INDEX_URL: {e}") from e


def _load_index_from_disk() -> dict:
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def build_index() -> dict:
    ensure_index_present()
    if INDEX_FILE.exists():
        print("Loading existing index...")
        try:
            return _load_index_from_disk()
        except Exception as e:
            print(f"Failed to load index ({INDEX_FILE}): {e}")
            try:
                INDEX_FILE.unlink()
            except Exception:
                pass
            if INDEX_URL:
                print("Re-downloading index after load failure...")
                ensure_index_present()
                return _load_index_from_disk()
            raise

    if RESEARCH_DIR is None or OUTPUT_CSV is None:
        raise RuntimeError("Cannot build index: HEP_RESEARCH_DIR is not set.")
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV at '{OUTPUT_CSV}'.")

    print("Building index from scratch...")
    metadata = {}
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("renamed_filename"):
                metadata[row["renamed_filename"]] = row

    documents = []
    all_pdfs  = sorted(RESEARCH_DIR.glob("*.pdf"))
    for i, pdf_path in enumerate(all_pdfs, 1):
        filename = pdf_path.name
        meta     = metadata.get(filename, {})
        text     = extract_text(pdf_path)
        combined = " ".join(filter(None, [
            meta.get("full_title", ""), meta.get("authors", ""),
            meta.get("keywords", ""), meta.get("abstract_summary", ""), text
        ]))
        documents.append({"filename": filename, "path": str(pdf_path),
                          "text": text, "combined": combined, "meta": meta, "embedding": None})
        if i % 50 == 0:
            print(f"  Processed {i}/{len(all_pdfs)}...")

    vocab = build_vocab(documents)
    for i, doc in enumerate(documents, 1):
        doc["embedding"] = embed(doc["combined"], vocab)
        if i % 50 == 0:
            print(f"  Embedded {i}/{len(documents)}...")

    index = {"documents": documents, "vocab": vocab}
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    return index


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
    records      = _scroll_by_category(qdrant, category, limit=2000)
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
            model="claude-opus-4-5",
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

def generate_answer(query: str, results: list) -> dict:
    if not results:
        return {"answer": "No relevant documents found.", "citations": [], "contradictions": []}

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

    prompt = f"""You are a research assistant for Hope Education Project, an anti-trafficking NGO focused on Ghana and West Africa.

Answer the following research question using ONLY the provided sources.
- Write in clear analytical prose
- Cite sources using [number] notation inline
- Be specific about geographies, statistics, and findings
- Distinguish between trafficking and smuggling where relevant
- Note gaps or contradictions in the evidence
- Maximum 4 paragraphs

Research question: {query}

Sources:
{context}

Answer:"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    answer_text = message.content[0].text.strip()

    # Phase 3: run evidence strength and contradiction detection in parallel
    # (contradiction detection is a second Claude call - runs after answer generation)
    evidence_strength = compute_evidence_strength(citations, answer_text)
    for c in citations:
        c["evidence_strength"] = evidence_strength.get(c["index"], 1)

    contradictions = detect_contradictions(query, results)

    return {
        "answer":        answer_text,
        "citations":     citations,
        "contradictions": contradictions,
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────

def get_index():
    raise RuntimeError("Legacy TF-IDF index is disabled; semantic search uses Qdrant only.")


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
        records, _ = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=filt,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )

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
    }), 200


@app.route("/rebuild-index", methods=["POST"])
def rebuild():
    return jsonify({"error": "Legacy TF-IDF rebuild is disabled. Use build_embeddings.py + Qdrant."}), 410


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
