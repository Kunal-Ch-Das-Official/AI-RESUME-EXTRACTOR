# ResumeCraft Import — Audit and Verification Report

## Scope and contract checks

- Inspected the uploaded frontend archive, primary TypeScript backend archive, Python importer archive, and sample-resume archive.
- Confirmed the frontend uploads a multipart field named `file`, expects `success`, `document_id`, and `resume` in the response, and redirects to `/resume-builder/<document_id>`.
- Confirmed the primary backend uses `/api/v1/resume/upload-resume` for its own create endpoint and stores resume data in the `fresh_resume_information` collection; the importer uses the separate `/api/v1/resumes/import` endpoint and defaults to the same collection.
- No frontend or primary-backend source files were modified.
- Existing TypeScript contract test remains part of the importer test suite.

## Changes implemented

1. Generalized deterministic skill parsing for comma-, semicolon-, pipe-, and bullet-separated skill values; removed the requirement for specific `Software:` / `Standards Knowledge:` labels.
2. Merged AI output with deterministic extraction so non-empty local fields and records missing from a partial AI result are retained. AI values win for non-empty scalar conflicts; list items and mapping records are preserved.
3. Allowed empty email values where the source resume does not contain an email, preventing an otherwise useful resume from failing Pydantic validation.
4. Normalized common bare LinkedIn/GitHub/domain URLs to HTTP(S) URLs during local extraction and validation.
5. Removed a resume-specific certification-name special case and a hardcoded Kolkata address inference.
6. Added OCR for individual text-empty PDF pages, including mixed PDFs that contain both selectable text and scanned pages.
7. Added regression tests for skill splitting, missing email, URL normalization, certificate generalization, and preservation of local records when AI output is incomplete.
8. Added `scripts/evaluate_samples.py` for repeatable, privacy-conscious processing/coverage checks.

## Test results

- Baseline: **6 passed, 1 failed**. The failure was the existing deterministic skills test.
- After changes: **14 tests passed** (including newly added regressions).
- `compileall`: completed successfully.
- Ruff was not run because the Ruff module is not installed in the execution environment.
- MongoDB-backed integration and live AI-provider tests were not run because this environment did not provide the complete service dependencies/credentials. No live AI extraction or persistence is claimed.

## Sample-resume evaluation

The supplied archive contained **47 PDF resumes**. All **47 / 47** were readable by the local PDF extraction + deterministic parser in this environment.

Coverage indicators from the local parser (not accuracy measurements):

- Non-empty name: 47 / 47
- Non-empty email: 47 / 47
- Non-default position: 42 / 47
- At least one skill: 36 / 47
- At least one education record: 35 / 47
- At least one work-experience record: 28 / 47
- At least one project: 17 / 47

The sample archive did not include annotated expected outputs. These figures are **not** precision, recall, or field accuracy. Empty sections may legitimately be absent from a resume, and some local-parser misses remain; the AI path was not exercised against these samples here. A ground-truth annotation set and credentialed end-to-end run are still needed before making extraction-accuracy claims.

## Remaining limitations and deployment notes

- No 100% extraction accuracy is claimed. Complex columns, unusual headings, dates, and multiple records still need annotated regression coverage.
- RAG/knowledge ingestion and MongoDB persistence need verification in an environment with configured dependencies, credentials, and the production-equivalent database.
- Configure `GEMINI_API_KEY` and/or `OPENROUTER_API_KEY`, plus the appropriate MongoDB settings, before enabling those paths.
- OCR of scanned pages requires the Tesseract executable installed on the host, in addition to the Python package.
- Keep `.env` out of source control and configure secrets through deployment environment variables.
- The output model retains the existing ResumeCraft field names and route contract; no external API contract was intentionally changed.

## Follow-up repair: section completeness and upload types (2026-10-09)

- Added a `rawSections` mapping to the validated resume payload. It retains each recognized section heading and its original section text, including unfamiliar section names that do not map to the normalized schema.
- Added conservative dynamic heading recognition for unknown resume-specific headings while avoiding common false headings such as bullets, date lines, and comma-separated job entries.
- Fixed false phone extraction from date ranges such as `1999-2002`; phone candidates must contain at least 10 digits.
- Restricted resume uploads at the API route to PDF and DOCX. Plain text and image uploads are rejected; legacy binary `.doc` files receive a conversion message.
- Added regression coverage for the functional resume structure and supported file types.
- Verification: `pytest -q` -> 16 passed; `python -m compileall -q app scripts tests` passed. On `functionalsample.pdf`, the deterministic extractor preserved all five source sections (Career Summary, Adult Care Experience, Childcare Experience, Employment History, Education), all three employment date ranges in the employment section, both degrees in education, and no fabricated phone number.
- Limitation: dynamic section detection is heuristic and cannot guarantee perfect heading segmentation for every layout. OCR accuracy depends on local Tesseract availability; live AI-provider and MongoDB integration were not tested in this environment.
