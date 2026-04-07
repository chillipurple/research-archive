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
- **`HEP_INDEX_URL`**: HTTPS URL to `_vector_index.pkl` in your bucket
- **`HEP_INDEX_FILE`** (optional): local path to store the downloaded index (default: `./data/_vector_index.pkl`)

Optional (local-only):

- **`HEP_RESEARCH_DIR`**: path to your Google Drive "Research" folder (required only for `/rebuild-index`)

### Updating when new PDFs are added

1. On your Mac, add PDFs to Google Drive and rebuild the index (locally).
2. Upload the new `_vector_index.pkl` to your bucket (same key/path if you want the URL to stay stable).
3. Restart the Railway service (or redeploy) so it downloads the latest index at startup.

