"""
Single-PDF ingest pipeline: Cloudflare R2, Google Drive, Qdrant embeddings.
Reuses chunking and embedding helpers from build_embeddings.py.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import boto3
import voyageai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from build_embeddings import (
    CHUNK_WORDS,
    EMBED_BATCH,
    OVERLAP_WORDS,
    UPSERT_BATCH,
    QDRANT_COLLECTION,
    embed_texts,
    chunk_words,
    ensure_collection,
    pdf_words_with_pages,
    load_metadata,
)

GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _qdrant_api_key() -> str:
    key = (
        os.environ.get("QDRANT_API_KEY")
        or os.environ.get("HEP_QDRANT_API_KEY")
        or os.environ.get("QDRANT_KEY")
        or ""
    ).strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def metadata_row_for_filename(filename: str) -> dict:
    csv_env = os.environ.get("HEP_RESEARCH_CSV", "").strip()
    if csv_env:
        p = Path(csv_env).expanduser()
    else:
        rd = os.environ.get("HEP_RESEARCH_DIR", "").strip()
        if not rd:
            return {}
        p = Path(rd).expanduser() / "_research_index.csv"
    if not p.exists():
        return {}
    meta = load_metadata(p)
    return dict(meta.get(filename, {}))


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"].strip(),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"].strip(),
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"].strip(),
    )


def upload_r2(local_path: Path, filename: str) -> str:
    bucket = os.environ["R2_BUCKET_NAME"].strip()
    key = f"pdfs/{filename}"
    cli = _r2_client()
    cli.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def upload_google_drive(local_path: Path, filename: str) -> str:
    raw = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"].strip()
    info = json.loads(raw)
    folder_id = os.environ["GOOGLE_DRIVE_FOLDER_ID"].strip()
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=GOOGLE_DRIVE_SCOPES
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    body = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), mimetype="application/pdf", resumable=True)
    created = (
        service.files()
        .create(body=body, media_body=media, fields="id", supportsAllDrives=True)
        .execute()
    )
    return created.get("id", "")


def delete_qdrant_chunks_for_filename(qdrant: QdrantClient, filename: str) -> None:
    try:
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="filename",
                            match=qm.MatchValue(value=filename),
                        )
                    ]
                )
            ),
        )
    except Exception:
        pass


def embed_pdf_to_qdrant(
    local_path: Path,
    filename: str,
    meta: dict,
    qdrant: QdrantClient,
    voy: voyageai.Client,
) -> int:
    title = (meta.get("full_title") or filename).strip()
    authors = (meta.get("authors") or "Unknown").strip()
    year = (meta.get("year") or "n.d.").strip()
    category = (meta.get("category") or "").strip()
    keywords = (meta.get("keywords") or "").strip()
    abstract_summary = (meta.get("abstract_summary") or "").strip()

    words = pdf_words_with_pages(local_path)
    chunks = chunk_words(words, CHUNK_WORDS, OVERLAP_WORDS)

    probe = embed_texts(voy, ["probe"])[0]
    vector_size = len(probe)
    ensure_collection(qdrant, QDRANT_COLLECTION, vector_size=vector_size)

    delete_qdrant_chunks_for_filename(qdrant, filename)

    if not chunks:
        return 0

    embed_inputs: List[str] = []
    payloads: List[dict] = []
    ids: List[str] = []

    for c in chunks:
        header = "\n".join(
            [
                f"Title: {title}",
                f"Authors: {authors}",
                f"Year: {year}",
                f"Category: {category}",
                f"Keywords: {keywords}",
                f"Abstract: {abstract_summary}",
                f"Filename: {filename}",
                f"Pages: {c.page_start}-{c.page_end}",
                f"Chunk: {c.chunk_index}",
            ]
        ).strip()
        text_for_embedding = f"{header}\n\nChunk text:\n{c.text}"
        embed_inputs.append(text_for_embedding)
        payloads.append(
            {
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
            }
        )
        ids.append(str(uuid.uuid4()))

    upsert_buffer: List[qm.PointStruct] = []
    total = 0
    for start in range(0, len(embed_inputs), EMBED_BATCH):
        end = min(len(embed_inputs), start + EMBED_BATCH)
        batch_inputs = embed_inputs[start:end]
        batch_payloads = payloads[start:end]
        batch_ids = ids[start:end]
        vectors = embed_texts(voy, batch_inputs)
        for pid, vec, pl in zip(batch_ids, vectors, batch_payloads):
            upsert_buffer.append(qm.PointStruct(id=pid, vector=vec, payload=pl))
        if len(upsert_buffer) >= UPSERT_BATCH:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=upsert_buffer)
            total += len(upsert_buffer)
            upsert_buffer = []
    if upsert_buffer:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=upsert_buffer)
        total += len(upsert_buffer)
    return total


def ingest_pdf(local_path: Path, filename: str) -> Dict[str, Any]:
    """
    Upload to R2, Google Drive, then embed to Qdrant. Returns step log and chunk count.
    """
    steps: List[Dict[str, Any]] = []
    meta = metadata_row_for_filename(filename)

    try:
        r2_uri = upload_r2(local_path, filename)
        steps.append({"name": "r2", "ok": True, "detail": r2_uri})
    except Exception as e:
        steps.append({"name": "r2", "ok": False, "detail": str(e)})
        return {"ok": False, "steps": steps, "error": str(e), "chunks_upserted": 0}

    try:
        fid = upload_google_drive(local_path, filename)
        steps.append({"name": "google_drive", "ok": True, "detail": f"Drive file id: {fid}"})
    except Exception as e:
        steps.append({"name": "google_drive", "ok": False, "detail": str(e)})
        return {"ok": False, "steps": steps, "error": str(e), "chunks_upserted": 0}

    if not os.environ.get("VOYAGE_API_KEY", "").strip():
        msg = "VOYAGE_API_KEY is not set"
        steps.append({"name": "qdrant", "ok": False, "detail": msg})
        return {"ok": False, "steps": steps, "error": msg, "chunks_upserted": 0}

    url = os.environ.get("QDRANT_URL", "").strip()
    key = _qdrant_api_key()
    if not url or not key:
        msg = "QDRANT_URL and QDRANT_API_KEY must be set"
        steps.append({"name": "qdrant", "ok": False, "detail": msg})
        return {"ok": False, "steps": steps, "error": msg, "chunks_upserted": 0}

    try:
        qdrant = QdrantClient(url=url, api_key=key, timeout=120)
        voy = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"].strip())
        n = embed_pdf_to_qdrant(local_path, filename, meta, qdrant, voy)
        steps.append({"name": "qdrant", "ok": True, "detail": f"{n} chunks upserted"})
        return {"ok": True, "steps": steps, "error": None, "chunks_upserted": n}
    except Exception as e:
        steps.append({"name": "qdrant", "ok": False, "detail": str(e)})
        return {"ok": False, "steps": steps, "error": str(e), "chunks_upserted": 0}


def upload_config_ok() -> bool:
    try:
        return bool(
            os.environ.get("R2_ENDPOINT_URL", "").strip()
            and os.environ.get("R2_ACCESS_KEY_ID", "").strip()
            and os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
            and os.environ.get("R2_BUCKET_NAME", "").strip()
            and os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
            and os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "").strip()
        )
    except Exception:
        return False
