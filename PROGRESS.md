# Audit progress

- [x] Inventory all four uploaded archives and inspect repository structures.
- [x] Inspect frontend upload request/response expectations.
- [x] Inspect primary backend route and schema/collection contract.
- [x] Run baseline tests and reproduce the deterministic skill parsing failure.
- [x] Repair generic skill splitting and missing-email validation.
- [x] Preserve deterministic extraction when AI output is partial.
- [x] Remove resume-specific certification/address assumptions.
- [x] OCR text-empty pages in mixed PDFs.
- [x] Add regression tests and sample-coverage script.
- [x] Run automated tests and compile checks.
- [x] Process the 47 supplied PDFs with local extraction and document coverage limitations.
- [ ] Run live AI/MongoDB integration tests with production-equivalent credentials and services.
- [ ] Annotate representative sample resumes to measure field-level precision and recall.

Next step: configure the service environment, run the importer and integration tests against a disposable database, then evaluate against independently annotated expected outputs.

## Follow-up section-preservation fix (2026-10-09)
- [x] Preserve all detected source sections as `rawSections` text, including unrecognized headings.
- [x] Prevent employment date ranges from being classified as phone numbers.
- [x] Restrict import API to PDF and DOCX only; reject images and TXT.
- [x] Add functional-resume and file-type regression tests.
- [x] Run tests (16 passed) and Python compilation.
- [ ] Verify live Gemini/OpenRouter calls and MongoDB persistence using deployment credentials.
- [ ] Run held-out, manually labeled evaluation across the entire sample set.
