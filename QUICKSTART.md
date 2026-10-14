# Resume Importer v4 — quick start

This version imports PDFs, DOCX, TXT, and common image files without requiring
Gemini, MongoDB, a vector index, or a network connection. It stores imports in
`data/resumes.sqlite3` by default. Configure MongoDB only when you want shared
cloud storage; configure Gemini only for optional future enrichment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8080/docs`, then use `POST /api/v1/resumes/import` to
upload a resume. Scanned PDFs and images need the local Tesseract executable:

```bash
brew install tesseract
```

Every import stores the original extracted text, chunks, source evidence, and
an audit record. `needs_review` is intentionally true: no parser or LLM can
truthfully promise 100% accuracy for arbitrary resume layouts. Your UI should
let the user review and amend the returned fields before generating the final
resume.

## AI outage fallback

Create a free API key at [OpenRouter](https://openrouter.ai/keys) and set
`OPENROUTER_API_KEY` in `.env`.  The configured fallback model is
`openrouter/free`. If Gemini returns an unavailable/rate-limit error, the same
structured request is sent to OpenRouter. If both providers fail, the import is
still saved with deterministic extraction and `needs_review=true`.

Free API capacity is deliberately not treated as a production SLA. OpenRouter
currently documents a 50-request/day free limit. For a real production SLA,
replace `OPENROUTER_MODEL` with a paid, pinned model and keep the same failover
architecture.

## Production notes

- Keep `.env` out of source control and rotate any keys that have ever been
  committed or shared.
- Set `DATABASE_MODE=local` to force SQLite, or use `auto` with `MONGODB_URI`
  for MongoDB with local fallback.
- The legacy `/knowledge/ingest` endpoint still requires MongoDB and Gemini.
  It is not part of the normal import path.


### Supported resume uploads

Resume import accepts PDF (`.pdf`) and Word Open XML (`.docx`) files only. Image files, plain text files, and legacy Word `.doc` files are rejected. For scanned PDFs, Tesseract OCR must be installed. The importer preserves a `rawSections` mapping containing each detected section heading and its original text, including sections that cannot be mapped to a normalized resume field.
