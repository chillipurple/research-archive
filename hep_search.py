#!/usr/bin/env python3
"""
HEP Research Library - Search Application
A Flask web app for querying the HEP research PDF library.
Uses Claude API for answer generation and TF-IDF for search.
"""

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

# Local (Mac) research folder configuration (only required when rebuilding index).
RESEARCH_DIR_ENV = os.environ.get("HEP_RESEARCH_DIR")
RESEARCH_DIR = Path(RESEARCH_DIR_ENV) if RESEARCH_DIR_ENV else None
OUTPUT_CSV = (RESEARCH_DIR / "_research_index.csv") if RESEARCH_DIR else None

# Cloud/runtime index configuration (recommended for deployment).
# Provide HEP_INDEX_URL to download the pre-built pickle at startup (Railway + R2, per handover Section 6).
INDEX_URL = os.environ.get("HEP_INDEX_URL", "").strip() or None
INDEX_FILE = Path(os.environ.get("HEP_INDEX_FILE", str(BASE_DIR / "data" / "_vector_index.pkl"))).resolve()

# Semantic search (Qdrant + Voyage)


def _clean_env_str(val: str | None) -> str:
    """Match pdf_ingest: strip BOM and surrounding quotes (Railway / pasted secrets)."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


VOYAGE_API_KEY = _clean_env_str(os.environ.get("VOYAGE_API_KEY"))
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3").strip() or "voyage-3"
QDRANT_URL = _clean_env_str(os.environ.get("QDRANT_URL"))


def _read_qdrant_api_key() -> str:
    """
    Read Qdrant API key from env, supporting common legacy aliases.
    """
    key = (
        _clean_env_str(os.environ.get("QDRANT_API_KEY"))
        or _clean_env_str(os.environ.get("HEP_QDRANT_API_KEY"))
        or _clean_env_str(os.environ.get("QDRANT_KEY"))
        or ""
    )
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


QDRANT_API_KEY = _read_qdrant_api_key()
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "hep_research").strip() or "hep_research"

# Public PDFs on Cloudflare R2: {HEP_PDF_BASE_URL}/{filename}
HEP_PDF_BASE_URL = os.environ.get(
    "HEP_PDF_BASE_URL",
    "https://pub-40a226eaa06b4f59988b1d1213d57d9c.r2.dev/pdfs",
).strip().rstrip("/")

client = anthropic.Anthropic()
app = Flask(__name__)

_max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "100"))
app.config["MAX_CONTENT_LENGTH"] = _max_upload_mb * 1024 * 1024

try:
    from pdf_ingest import ingest_pdf, upload_config_detail, upload_config_ok
except ImportError:  # pragma: no cover
    ingest_pdf = None  # type: ignore

    def upload_config_detail() -> dict:  # type: ignore
        return {
            "ok": False,
            "checks": [
                {
                    "id": "pdf_ingest_import",
                    "name": "pdf_ingest module",
                    "ok": False,
                    "detail": "not importable (install boto3, google-api-python-client, etc.)",
                }
            ],
        }

    def upload_config_ok() -> bool:  # type: ignore
        return False

# ── HTTP Basic Auth (optional: set AUTH_USERNAME + AUTH_PASSWORD in production) ─

AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "").strip()
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "").strip()
AUTH_ENABLED = bool(AUTH_USERNAME and AUTH_PASSWORD)


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

_voy = None
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
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY (or HEP_QDRANT_API_KEY) must be set")
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return _qdrant


def embed_query(text: str) -> list:
    voy = get_voyage_client()
    resp = voy.embed(texts=[text], model=VOYAGE_MODEL)
    return resp.embeddings[0]


def pdf_url_for_filename(filename: str) -> str:
    """HTTPS URL to the PDF on R2: base URL + path-safe filename."""
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
                # Some CDNs return 403 for unknown/empty user agents.
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
            "Set HEP_INDEX_URL to a downloadable _vector_index.pkl for cloud deployment, "
            "or set HEP_RESEARCH_DIR to rebuild locally."
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
    # Prefer pre-built index (local or downloaded) when present.
    ensure_index_present()
    if INDEX_FILE.exists():
        print("Loading existing index...")
        try:
            return _load_index_from_disk()
        except Exception as e:
            # If the wrong file was downloaded (e.g. URL misconfigured), recover by
            # deleting and re-downloading from HEP_INDEX_URL.
            print(f"Failed to load index from disk ({INDEX_FILE}): {e}")
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
        raise RuntimeError(
            "Cannot build index: HEP_RESEARCH_DIR is not set. "
            "For cloud deployment, set HEP_INDEX_URL instead."
        )
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV at '{OUTPUT_CSV}'.")

    print("Building index from scratch - this will take several minutes...")

    metadata = {}
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("renamed_filename"):
                metadata[row["renamed_filename"]] = row

    documents = []
    all_pdfs = sorted(RESEARCH_DIR.glob("*.pdf"))
    print(f"Processing {len(all_pdfs)} PDFs...")

    for i, pdf_path in enumerate(all_pdfs, 1):
        filename = pdf_path.name
        meta = metadata.get(filename, {})
        text = extract_text(pdf_path)
        combined = " ".join(filter(None, [
            meta.get("full_title", ""),
            meta.get("authors", ""),
            meta.get("keywords", ""),
            meta.get("abstract_summary", ""),
            text
        ]))
        documents.append({
            "filename": filename,
            "path": str(pdf_path),
            "text": text,
            "combined": combined,
            "meta": meta,
            "embedding": None
        })
        if i % 50 == 0:
            print(f"  Processed {i}/{len(all_pdfs)}...")

    print("Building vocabulary...")
    vocab = build_vocab(documents)
    print(f"Vocabulary size: {len(vocab)} terms")

    print("Computing embeddings...")
    for i, doc in enumerate(documents, 1):
        doc["embedding"] = embed(doc["combined"], vocab)
        if i % 50 == 0:
            print(f"  Embedded {i}/{len(documents)}...")

    index = {"documents": documents, "vocab": vocab}
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)

    print(f"Index built and saved. {len(documents)} documents indexed.")
    return index


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, index: dict, category: str = "All", top_k: int = 8) -> list:
    vocab     = index["vocab"]
    documents = index["documents"]
    query_vec = embed(query, vocab)

    results = []
    for doc in documents:
        if category and category != "All":
            if doc["meta"].get("category", "") != category:
                continue
        score = cosine_similarity(query_vec, doc["embedding"])
        results.append((score, doc))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def semantic_search(query: str, category: str = "All", top_k: int = 8) -> list:
    """
    Returns Qdrant hits as (score, payload_dict).
    """
    qdrant = get_qdrant_client()
    vec = embed_query(query)

    filt = None
    if category and category != "All":
        filt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="category",
                    match=qm.MatchValue(value=category),
                )
            ]
        )

    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=filt,
    )
    hits = response.points
    out = []
    for h in hits:
        out.append((float(h.score), dict(h.payload or {})))
    return out

# ── Answer Generation ─────────────────────────────────────────────────────────

def generate_answer(query: str, results: list) -> dict:
    if not results:
        return {"answer": "No relevant documents found.", "citations": []}

    context_parts = []
    citations     = []
    doc_url_template = os.environ.get("HEP_DOC_URL_TEMPLATE", "").strip()

    for i, (score, doc) in enumerate(results, 1):
        # `doc` here is a Qdrant payload dict produced by build_embeddings.py
        title = doc.get("title") or doc.get("filename") or "Unknown title"
        authors = doc.get("authors") or "Unknown"
        year = doc.get("year") or "n.d."
        filename = doc.get("filename") or ""
        category = doc.get("category") or ""
        chunk_index = doc.get("chunk_index")
        page_start = doc.get("page_start")
        page_end = doc.get("page_end")
        chunk_text = doc.get("text") or ""

        ref_bits = []
        if page_start and page_end:
            ref_bits.append(f"pp. {page_start}-{page_end}")
        if chunk_index is not None:
            ref_bits.append(f"chunk {chunk_index}")
        ref = ("; " + ", ".join(ref_bits)) if ref_bits else ""

        excerpt = chunk_text[:2000]
        context_parts.append(f"[{i}] {title} ({authors}, {year}){ref}\n{excerpt}")

        citation = {
            "index": i,
            "title": title,
            "authors": authors,
            "year": year,
            "filename": filename,
            "category": category,
            "relevance_score": round(score, 3),
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
            "pdf_url": pdf_url_for_filename(filename),
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

    return {
        "answer":    message.content[0].text.strip(),
        "citations": citations
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────

_index = None

def get_index():
    raise RuntimeError("Legacy TF-IDF index is disabled; semantic search uses Qdrant only.")


@app.route("/")
def home():
    # In Qdrant mode, document count is not loaded from a local pickle index.
    doc_count = "Qdrant"
    logo_url = os.environ.get("HEP_LOGO_URL", "").strip() or "/logo"
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
        # Semantic search via Qdrant + Voyage
        hits = semantic_search(query, category=category, top_k=8)
        return jsonify(generate_answer(query, hits))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/health")
def health():
    # Keep health checks fast and non-blocking: no network calls here.
    return jsonify({
        "ok": True,
        "qdrant_configured": bool(QDRANT_URL and QDRANT_API_KEY),
        "qdrant_lazy_connect": True,
    }), 200


@app.route("/rebuild-index", methods=["POST"])
def rebuild():
    return jsonify({"error": "Legacy TF-IDF rebuild is disabled. Use build_embeddings.py + Qdrant."}), 410


@app.route("/admin")
def admin_page():
    logo_url = os.environ.get("HEP_LOGO_URL", "").strip() or "/logo"
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
        qc = get_qdrant_client()
        cnt = qc.count(collection_name=QDRANT_COLLECTION, exact=True).count
        return jsonify({"ok": True, "total_chunks": cnt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "total_chunks": None}), 200


_ADMIN_DEBUG_ENV_KEYS = (
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "HEP_QDRANT_API_KEY",
    "VOYAGE_API_KEY",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET_NAME",
    "R2_ENDPOINT_URL",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_DRIVE_FOLDER_ID",
)


@app.route("/admin/api/debug")
def admin_debug():
    """
    Returns True/False per key: whether a non-empty value exists after the same cleaning
    used for secrets (strip, BOM, surrounding quotes). No secret values are returned.
    """
    return jsonify(
        {k: bool(_clean_env_str(os.environ.get(k))) for k in _ADMIN_DEBUG_ENV_KEYS}
    )


@app.route("/admin/api/upload-debug")
def admin_upload_debug():
    """
    Detailed pass/fail for each upload_config_ok check (same logic as ingest).
    Does not expose secret values.
    """
    detail = upload_config_detail()
    return jsonify(detail)


@app.route("/admin/api/upload", methods=["POST"])
def admin_upload():
    if ingest_pdf is None:
        return jsonify({"ok": False, "error": "pdf_ingest module not available", "steps": []}), 500
    if not upload_config_ok():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Missing R2 or Google Drive environment variables",
                    "steps": [],
                }
            ),
            400,
        )
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file provided", "steps": []}), 400
    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "Only PDF files are allowed", "steps": []}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = Path(tmp.name)
    try:
        tmp.close()
        f.save(tmp_path)
        result = ingest_pdf(tmp_path, filename)
        code = 200 if result.get("ok") else 500
        return jsonify(
            {
                "ok": bool(result.get("ok")),
                "steps": result.get("steps", []),
                "error": result.get("error"),
                "chunks_upserted": result.get("chunks_upserted", 0),
            }
        ), code
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HEP Research Library")
    print("=" * 50)
    print("Starting API server...")
    print("Open http://localhost:5000 when ready.")
    print()
    print("Server running at http://localhost:5000")
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
