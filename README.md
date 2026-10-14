# Resume Importer — RAG Edition

A production-oriented semantic resume importer for Resume Craft.

## What this version changes

This is **not RAG-only**. Resume importing is an information-extraction problem, so the pipeline is hybrid:

1. Parse the uploaded document with layout/page awareness.
2. OCR scanned/image documents when needed.
3. Create semantic chunks with source/page references.
4. Retrieve relevant resume-schema knowledge and labeled examples from MongoDB Atlas Vector Search.
5. Ask Gemini for structured extraction **with evidence references**.
6. Run deterministic validation/invariants.
7. Run a separate verification pass against the source chunks.
8. Calculate confidence and mark uncertain fields for review.
9. Store the normalized resume plus audit/evidence data.

RAG is used to provide the model with:
- canonical meanings of Resume Craft fields;
- synonyms and section concepts;
- examples of how unusual headings map to canonical fields;
- edge cases and extraction rules;
- previously approved, sanitized examples.

The model is NOT allowed to invent data.

## Important: "bulletproof"

No LLM-based importer can honestly be called 100% bulletproof. The correct production design is:

**extract → ground → validate → verify → confidence → review if uncertain**

The importer should never silently turn uncertainty into fake resume data.

## Architecture

```text
                         ┌─────────────────────────┐
                         │ Resume Craft / Next.js  │
                         └────────────┬────────────┘
                                      │ multipart upload
                                      ▼
                         ┌─────────────────────────┐
                         │ FastAPI Import API      │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │ Document Parser         │
                         │ PDF / DOCX / TXT / OCR  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Layout-aware chunks      │
                         │ page + block + text      │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴──────────────────┐
                    │                                    │
                    ▼                                    ▼
        ┌─────────────────────┐              ┌─────────────────────┐
        │ Gemini Embeddings   │              │ Gemini Extraction   │
        │ gemini-embedding-2  │              │ structured output   │
        └──────────┬──────────┘              └──────────┬──────────┘
                   │                                    │
                   ▼                                    │
        ┌──────────────────────────────┐                 │
        │ MongoDB Atlas Vector Search  │◄────────────────┘
        │ schema + examples + rules    │
        └──────────────┬───────────────┘
                       │ retrieved grounding
                       ▼
              ┌──────────────────────┐
              │ Evidence Resolver    │
              │ field → source refs  │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ Deterministic QA     │
              │ dates, URLs, emails, │
              │ duplicates, flags    │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ Gemini Verifier       │
              │ source vs extraction │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ Confidence / Review  │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ MongoDB Resume DB    │
              │ resume + audit       │
              └──────────────────────┘
```

## Why RAG helps

Suppose a resume says:

- `Qualifications` → education
- `Academic Journey` → education
- `Professional Background` → work experience
- `Selected Work` → projects
- `Technical Arsenal` → skills
- `Research Output` → publications
- `Honors & Distinctions` → achievements

A static parser has to know these terms ahead of time.

With RAG, the extraction model receives relevant canonical concepts and examples retrieved from the knowledge base. The vocabulary can therefore grow without changing Python code.

## What should be put into the RAG knowledge base?

### Good knowledge

- Resume Craft field definitions.
- Canonical section meanings.
- Synonym examples.
- Ambiguous-heading examples.
- Extraction edge cases.
- Sanitized/approved sample resume chunks.
- Human-reviewed corrections.

### Do NOT blindly put every real user's resume into the knowledge base.

Real resumes contain personal information. Prefer dummy/anonymized resumes for the reusable knowledge base. Real resumes can be used as an evaluation set or imported as normal user documents.

## Requirements

- Python 3.11+ recommended.
- MongoDB Atlas with Vector Search enabled.
- Gemini API key.
- Tesseract only if image/scanned resume OCR is required.

### Environment

Copy `.env.example` to `.env`.

```env
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=resume_craft
MONGODB_COLLECTION=resumes

MONGODB_KB_COLLECTION=resume_import_knowledge
MONGODB_VECTOR_INDEX=resume_knowledge_vector

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=768

MAX_UPLOAD_MB=10
MIN_CONFIDENCE=0.78
REVIEW_CONFIDENCE=0.88
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If OCR is required on macOS:

```bash
brew install tesseract
```

## Create the Vector Search index

After `.env` is configured:

```bash
python scripts/create_vector_index.py
```

The index is configured for the embedding dimension in `GEMINI_EMBEDDING_DIMENSIONS`.

## Seed the knowledge base

```bash
python scripts/seed_knowledge.py
```

This creates canonical field definitions, synonyms, edge cases, and extraction rules.

## Add your own approved examples

Use:

```bash
python scripts/ingest_knowledge_resume.py /absolute/path/to/dummy-resume.pdf
```

Use dummy/anonymized resumes for this reusable knowledge base.

## Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger:

```text
http://127.0.0.1:8080/docs
```

## Import a resume

```bash
curl -X POST \
  -F "file=@/absolute/path/to/resume.pdf" \
  http://127.0.0.1:8080/api/v1/resumes/import
```

The response contains:

- `document_id`
- normalized `resume`
- `confidence`
- `needs_review`
- `audit`
- extraction engine/version

## Important API design

Do not store only this:

```json
{
  "resume": { "...": "..." }
}
```

Store this as well:

```json
{
  "resume": {},
  "importAudit": {
    "confidence": 0.94,
    "needsReview": false,
    "fieldEvidence": {},
    "warnings": [],
    "retrievedKnowledge": []
  }
}
```

This is what makes the importer debuggable.

## Evaluation workflow

Once you provide your dummy resumes, create a small gold dataset:

```text
evaluation/
  resumes/
    resume-001.pdf
    resume-002.pdf
  expected/
    resume-001.json
    resume-002.json
```

Then compare:

- field presence
- section classification
- entry count
- dates
- URLs
- bullets
- technologies
- education
- work history

Do this before deployment.

## Production recommendation

Do not make every uncertain extraction immediately available to the user's resume editor.

Use:

```text
confidence >= 0.88
    -> accept

0.78 - 0.879
    -> accept but flag

< 0.78
    -> review/repair workflow
```

These thresholds are starting points, not magic constants. Tune them against your real evaluation set.

## Current Resume Craft schema

The importer keeps compatibility with the existing schema:

- basicInfo
- address
- contactInfo
- profileSummary
- educations
- workExperience
- skills
- languageProficiency
- hobbies
- projects
- certifications
- openSource
- publications
- awardsAndAchievements

The RAG layer does not change your frontend schema.

## Next step

Give this project your dummy resumes first. We should then build an evaluation set and tune the retrieval/extraction prompts against real formatting variation before using user resumes in production.


### Supported resume uploads

Resume import accepts PDF (`.pdf`) and Word Open XML (`.docx`) files only. Image files, plain text files, and legacy Word `.doc` files are rejected. For scanned PDFs, Tesseract OCR must be installed. The importer preserves a `rawSections` mapping containing each detected section heading and its original text, including sections that cannot be mapped to a normalized resume field.
