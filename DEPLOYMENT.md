## Deployment (Section 6 approach)

This project is designed to deploy using **Railway.app** with a **pre-built vector index** hosted in a cloud storage bucket (e.g. Cloudflare R2).

The cloud server does **not** need access to the PDFs in Google Drive for normal search traffic. PDFs are only needed to rebuild the index.

### What to upload to cloud storage (R2/S3)

- `_vector_index.pkl` (typically ~200–400MB)

Make it downloadable via an HTTPS URL.

### Railway configuration

- **Build**: Railway will install `requirements.txt`
- **Start command**: uses `Procfile` (Gunicorn)

Set these environment variables in Railway:

- **`ANTHROPIC_API_KEY`**: your Claude API key
- **`HEP_LOGO_URL`** (optional): public URL to the HEP logo PNG (recommended for cloud, since the logo is not committed)
- **`HEP_DOC_URL_TEMPLATE`** (optional): template for linking citations to documents, using `{filename}` placeholder (e.g. `https://example.com/docs/{filename}`)
- **`VOYAGE_API_KEY`**: Voyage AI API key
- **`QDRANT_URL`**: Qdrant endpoint URL (e.g. `http://204.168.162.233:6333`)
- **`QDRANT_API_KEY`**: Qdrant API key
- **`QDRANT_COLLECTION`** (optional): defaults to `hep_research`
- **`VOYAGE_MODEL`** (optional): defaults to `voyage-3`

Optional (local-only):

- **`HEP_RESEARCH_DIR`**: path to your Google Drive "Research" folder (required only for `/rebuild-index`)

### Updating when new PDFs are added

1. On your Mac, add PDFs to Google Drive and rebuild the index (locally).
2. Run `build_embeddings.py` to (re)embed chunks and upsert to Qdrant.
3. No Railway restart is required; Qdrant serves the latest vectors immediately.

