# Resume payload contract

The Python importer emits the same root-level resume object consumed by the
Node backend: `basicInfo`, `address`, `contactInfo`, `profileSummary`,
`educations`, `skills`, `languageProficiency`, optional resume sections, and
`templateName`.

The MongoDB collection defaults to `fresh_resume_information`, which is the
model name used by `src/models/resumeInfo.ts`. Import diagnostics are stored in
`_importMetadata` and deliberately excluded from the importer's GET response.

## Source of truth

The request shape is mirrored from `src/schema/resume.schema.ts`. Keep the
Python Pydantic model and its tests synchronized when that TypeScript file
changes.

## Existing TypeScript inconsistency to resolve

`src/schema/resume.schema.ts` represents pending multipart attachments as
`{ fileIndex, title }`; `src/types/resume.types.ts` and
`src/models/resumeInfo.ts` represent stored attachment details as
`{ title, url, key }`. These are two lifecycle representations, not identical
schemas. The importer never invents an attachment reference. After Node uploads
an attachment, it should convert the multipart reference to the stored-file
representation before saving through Mongoose.
