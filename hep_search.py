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
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
import anthropic
import fitz  # pymupdf

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

client = anthropic.Anthropic()
app = Flask(__name__)

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


# ── Index Building ────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
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


def build_index() -> dict:
    # Prefer pre-built index (local or downloaded) when present.
    ensure_index_present()
    if INDEX_FILE.exists():
        print("Loading existing index...")
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)

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


# ── Answer Generation ─────────────────────────────────────────────────────────

def generate_answer(query: str, results: list) -> dict:
    if not results:
        return {"answer": "No relevant documents found.", "citations": []}

    context_parts = []
    citations     = []

    for i, (score, doc) in enumerate(results, 1):
        meta    = doc["meta"]
        title   = meta.get("full_title", doc["filename"])
        authors = meta.get("authors", "Unknown")
        year    = meta.get("year", "n.d.")
        excerpt = " ".join(filter(None, [
            meta.get("abstract_summary", ""),
            doc["text"][:1500]
        ]))
        context_parts.append(f"[{i}] {title} ({authors}, {year})\n{excerpt}")
        citations.append({
            "index":           i,
            "title":           title,
            "authors":         authors,
            "year":            year,
            "filename":        doc["filename"],
            "category":        meta.get("category", ""),
            "relevance_score": round(score, 3)
        })

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
    global _index
    if _index is None:
        _index = build_index()
    return _index


@app.route("/")
def home():
    idx = get_index()
    return render_template("index.html", doc_count=len(idx["documents"]))


@app.route("/logo")
def serve_logo():
    return send_file(str(LOGO_PATH), mimetype="image/png")


@app.route("/search", methods=["POST"])
def search_route():
    data     = request.get_json()
    query    = data.get("query", "").strip()
    category = data.get("category", "All")
    if not query:
        return jsonify({"error": "No query provided"})
    try:
        idx     = get_index()
        results = search(query, idx, category=category)
        return jsonify(generate_answer(query, results))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/rebuild-index", methods=["POST"])
def rebuild():
    global _index
    if RESEARCH_DIR is None or OUTPUT_CSV is None:
        return jsonify({"error": "Rebuild requires local HEP_RESEARCH_DIR (Google Drive). Not available in cloud mode."})
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    _index = build_index()
    return jsonify({"status": "ok", "documents": len(_index["documents"])})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HEP Research Library")
    print("=" * 50)
    print("Loading index...")
    print("Open http://localhost:5000 when ready.")
    print()
    get_index()
    print()
    print("Server running at http://localhost:5000")
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
