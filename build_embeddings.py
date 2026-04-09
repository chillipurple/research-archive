#!/usr/bin/env python3
"""
Build semantic chunk embeddings and upsert into Qdrant.

Pipeline:
- Read PDF with PyMuPDF
- Chunk into ~500-word windows with ~50-word overlap (tracked with page references)
- Combine each chunk with document metadata from _research_index.csv
- Embed each chunk using Voyage AI (voyage-3)
- Upsert vectors + payload into Qdrant collection: hep_research
"""

from __future__ import annotations

import csv
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # pymupdf
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


RESEARCH_DIR = Path(os.environ.get("HEP_RESEARCH_DIR", "")).expanduser()
CSV_PATH = Path(os.environ.get("HEP_RESEARCH_CSV", "")).expanduser()

QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "hep_research").strip() or "hep_research"

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "").strip()
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3").strip() or "voyage-3"

CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "500"))
OVERLAP_WORDS = int(os.environ.get("OVERLAP_WORDS", "50"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))
UPSERT_BATCH = int(os.environ.get("UPSERT_BATCH", "256"))
SLEEP_ON_RATE_LIMIT_S = float(os.environ.get("SLEEP_ON_RATE_LIMIT_S", "10"))


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    page_start: int
    page_end: int
    text: str


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_metadata(csv_path: Path) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("renamed_filename") or "").strip()
            if not fn:
                continue
            meta[fn] = row
    return meta


def pdf_words_with_pages(pdf_path: Path) -> List[Tuple[str, int]]:
    """
    Returns a list of (word, page_number_1based) extracted from the PDF.
    Uses get_text("words") to preserve page attribution.
    """
    doc = fitz.open(str(pdf_path))
    out: List[Tuple[str, int]] = []
    try:
        for p in range(len(doc)):
            words = doc[p].get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
            page_num = p + 1
            for w in words:
                token = str(w[4]).strip()
                if token:
                    out.append((token, page_num))
    finally:
        doc.close()
    return out


def chunk_words(words: List[Tuple[str, int]], chunk_words: int, overlap_words: int) -> List[Chunk]:
    if chunk_words <= 0:
        return []
    if overlap_words >= chunk_words:
        overlap_words = max(0, chunk_words // 10)

    chunks: List[Chunk] = []
    step = max(1, chunk_words - overlap_words)
    chunk_index = 0
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        window = words[start:end]
        if not window:
            continue
        pages = [p for _, p in window]
        page_start = min(pages)
        page_end = max(pages)
        text = " ".join(w for w, _ in window).strip()
        if text:
            chunks.append(Chunk(chunk_index=chunk_index, page_start=page_start, page_end=page_end, text=text))
            chunk_index += 1
        if end >= len(words):
            break
    return chunks


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def embed_texts(voy: voyageai.Client, texts: List[str]) -> List[List[float]]:
    """
    Embeds texts using Voyage AI.
    Retries on transient failures.
    """
    while True:
        try:
            resp = voy.embed(texts=texts, model=VOYAGE_MODEL)
            return resp.embeddings
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "too many" in msg:
                print(f"Rate limit from Voyage; sleeping {SLEEP_ON_RATE_LIMIT_S}s...")
                time.sleep(SLEEP_ON_RATE_LIMIT_S)
                continue
            raise


def main() -> None:
    if not VOYAGE_API_KEY:
        _die("VOYAGE_API_KEY is not set")
    if not QDRANT_URL or not QDRANT_API_KEY:
        _die("QDRANT_URL and QDRANT_API_KEY must be set")
    if not RESEARCH_DIR or not RESEARCH_DIR.exists():
        _die("HEP_RESEARCH_DIR must point to the folder containing PDFs")
    if not CSV_PATH or not CSV_PATH.exists():
        _die("HEP_RESEARCH_CSV must point to _research_index.csv")

    meta = load_metadata(CSV_PATH)
    pdfs = sorted(RESEARCH_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs.")
    print(f"Metadata rows loaded: {len(meta)}")

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    voy = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Discover vector size using a single embedding.
    probe = embed_texts(voy, ["probe"])[0]
    vector_size = len(probe)
    ensure_collection(qdrant, QDRANT_COLLECTION, vector_size=vector_size)
    print(f"Using collection '{QDRANT_COLLECTION}' (vector size {vector_size}).")

    total_chunks = 0
    upsert_buffer: List[qm.PointStruct] = []

    for i, pdf_path in enumerate(pdfs, 1):
        filename = pdf_path.name
        m = meta.get(filename, {})

        title = (m.get("full_title") or filename).strip()
        authors = (m.get("authors") or "Unknown").strip()
        year = (m.get("year") or "n.d.").strip()
        category = (m.get("category") or "").strip()
        keywords = (m.get("keywords") or "").strip()
        abstract_summary = (m.get("abstract_summary") or "").strip()

        try:
            words = pdf_words_with_pages(pdf_path)
            chunks = chunk_words(words, CHUNK_WORDS, OVERLAP_WORDS)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] Failed reading/chunking {filename}: {e}")
            continue

        if not chunks:
            print(f"[{i}/{len(pdfs)}] No text chunks for {filename}; skipping")
            continue

        # Build embed texts with metadata context.
        embed_inputs: List[str] = []
        payloads: List[dict] = []
        ids: List[str] = []

        for c in chunks:
            header = "\n".join([
                f"Title: {title}",
                f"Authors: {authors}",
                f"Year: {year}",
                f"Category: {category}",
                f"Keywords: {keywords}",
                f"Abstract: {abstract_summary}",
                f"Filename: {filename}",
                f"Pages: {c.page_start}-{c.page_end}",
                f"Chunk: {c.chunk_index}",
            ]).strip()
            text_for_embedding = f"{header}\n\nChunk text:\n{c.text}"
            embed_inputs.append(text_for_embedding)

            payloads.append({
                "filename": filename,
                "title": title,
                "authors": authors,
                "year": year,
                "category": category,
                "keywords": keywords,
                "abstract_summary": abstract_summary,
                "chunk_index": c.chunk_index,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            })
            ids.append(str(uuid.uuid4()))

        # Embed and upsert in batches.
        print(f"[{i}/{len(pdfs)}] {filename}: {len(chunks)} chunks")
        for start in range(0, len(embed_inputs), EMBED_BATCH):
            end = min(len(embed_inputs), start + EMBED_BATCH)
            batch_inputs = embed_inputs[start:end]
            batch_payloads = payloads[start:end]
            batch_ids = ids[start:end]

            try:
                vectors = embed_texts(voy, batch_inputs)
            except Exception as e:
                print(f"  Embed failed for {filename} chunks {start}-{end}: {e}")
                continue

            for pid, vec, pl in zip(batch_ids, vectors, batch_payloads):
                upsert_buffer.append(qm.PointStruct(id=pid, vector=vec, payload=pl))

            if len(upsert_buffer) >= UPSERT_BATCH:
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=upsert_buffer)
                total_chunks += len(upsert_buffer)
                print(f"  Upserted {total_chunks} chunks total...")
                upsert_buffer = []

        # Upsert remaining for this document (keeps progress visible).
        if upsert_buffer:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=upsert_buffer)
            total_chunks += len(upsert_buffer)
            print(f"  Upserted {total_chunks} chunks total...")
            upsert_buffer = []

    print(f"Done. Total chunks upserted: {total_chunks}")


if __name__ == "__main__":
    main()

