## HEP Research Library — Technical Specification (April 2026)

### Overview

- **Purpose**: Browser-based research assistant for Hope Education Project (HEP) that searches a corpus of research PDFs (indexed offline) and uses Anthropic Claude to generate **cited** answers.
- **Core flow**: User submits question → server retrieves top \(k\) relevant sources from a pre-built TF‑IDF index → sends sources to Claude → returns answer + structured citations → UI formats citations + renders sources list.

### Repository

- **GitHub**: `chillipurple/research-archive`
- **Primary files**:
  - `hep_search.py`: Flask app, search/index logic, Anthropic call, deployment helpers
  - `templates/index.html`: single-page UI (HTML/CSS/JS)
  - `requirements.txt`: Python dependencies
  - `Procfile`: Gunicorn start command (for Railway)
  - `DEPLOYMENT.md`: deployment/runbook notes
  - `railway.toml`, `nixpacks.toml`, `.python-version`: Railway build configuration
  - `.gitignore`: excludes large/sensitive assets

### Architecture

- **Backend**: Python Flask (`hep_search.py`)
- **Frontend**: Jinja template (`templates/index.html`) with vanilla JS calling JSON endpoints
- **Search**:
  - Uses TF‑IDF style term-frequency vectors over a vocabulary built from the corpus.
  - Query embedding uses the same vocab; retrieval uses cosine similarity.
  - Optional category filter by metadata category field.
- **Answer generation**:
  - Sends top sources (title/authors/year + excerpt) to Claude.
  - Claude returns prose with inline citations `[n]`.
  - Server returns the answer string plus structured citations.

### Data model

- **Semantic vectors**: stored in **Qdrant** collection `hep_research`
- **PDF corpus**: Google Drive folder (local-only; required only to rebuild index)
- **Logo**: hosted in R2 for production; served locally when running on Mac

### Runtime index handling (cloud-first)

- **Query embedding**: Voyage AI `voyage-3`
- **Retrieval**: Qdrant top-k chunk search (optionally filtered by category)

### Local rebuild mode

- Run `build_embeddings.py` with access to Google Drive PDFs + `_research_index.csv` to (re)embed and upsert chunks to Qdrant.

### HTTP endpoints

- **`GET /`**
  - Renders `templates/index.html`
  - Provides `doc_count` (or `…` if index can’t be loaded yet)
  - Provides `logo_url` (defaults to `/logo` unless `HEP_LOGO_URL` is set)
- **`GET /logo`**
  - If `HEP_LOGO_URL` is set: **302 redirect** to the R2 logo URL
  - Else: serves `HEP_LOGO_PATH` if present; otherwise 404
- **`POST /search`**
  - Request JSON: `{ "query": string, "category": string }`
  - Response JSON: `{ "answer": string, "citations": [...] }`
  - Error JSON: `{ "error": string }`
- **`GET /health`**
  - **200**: `{ ok: true, index_loaded: true, documents_indexed: number }`
  - **503**: `{ ok: false, index_loaded: false, error: string }`

### Frontend behavior

- Uses `fetch('/search', ...)` to submit queries.
- Shows/hides a loading indicator during search.
- Formats citations without regex literals:
  - HTML-escapes the answer.
  - Converts `[n]` patterns into `<sup>[n]</sup>` via a small string scanner.
- Renders structured citations list (title/authors/year/category).

### Environment variables

**Required (production)**
- **`ANTHROPIC_API_KEY`**: Anthropic API key
- **`VOYAGE_API_KEY`**: Voyage AI API key
- **`QDRANT_URL`**: Qdrant URL (e.g. `http://204.168.162.233:6333`)
- **`QDRANT_API_KEY`**: Qdrant API key

**Recommended (production)**
- **`HEP_LOGO_URL`**: public HTTPS URL to the R2 logo PNG

**Optional**
- **`QDRANT_COLLECTION`**: defaults to `hep_research`
- **`VOYAGE_MODEL`**: defaults to `voyage-3`
- **`HEP_LOGO_PATH`**: local path to logo (used only if `HEP_LOGO_URL` not set)
- **`HEP_RESEARCH_DIR`**: local Google Drive research folder path (enables `/rebuild-index`)
- **`PORT`**: injected by Railway; also supported by `__main__` for local parity

### Deployment (current working setup)

- **Platform**: Railway
- **Builder**: Nixpacks (forced via `railway.toml`)
- **Process**: Gunicorn (`Procfile` / start command)
- **Index storage**: Cloudflare R2 public object (`r2.dev`)
- **Health check**: `GET /health`

### Runbook

- **Check health**:
  - Visit `/health` and confirm `ok=true` and expected `documents_indexed`.
- **Index update** (after adding PDFs locally):
  - Rebuild locally (with Google Drive available).
  - Upload new `_vector_index.pkl` to R2 (same object key to keep URL stable).
  - Redeploy Railway so it fetches the new file on boot (or change `HEP_INDEX_FILE` to force a fresh download).
- **Common failure modes**:
  - **403 on download**: usually public-access/URL issues; verify R2 object is public and URL is correct.
  - **Unpickle errors**: cached file is wrong/corrupt; app auto-deletes and re-downloads if `HEP_INDEX_URL` is valid.
  - **Logo missing**: ensure `HEP_LOGO_URL` is set; `/logo` will redirect to it.

### Constraints and future improvements

- **Search is keyword-based** (TF‑IDF). Consider semantic embeddings for improved relevance.
- **Index is large**. Consider private storage + signed URLs if index contents should not be public.
- **No auth** on endpoints. If exposing widely, add access control (especially if enabling remote rebuild).

