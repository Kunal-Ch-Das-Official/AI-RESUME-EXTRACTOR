"""Run a privacy-conscious coverage evaluation over a directory of resume files.

This script reports processing and parser coverage, not accuracy. To measure
precision/recall, annotate expected fields for a held-out sample set separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.file_extractor import extract_document
from app.services.local_extractor import extract_locally


def evaluate(root: Path) -> dict:
    supported = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in supported)
    failures = 0
    names = emails = positions = skills = education = experience = projects = certifications = 0

    for path in files:
        try:
            parsed = extract_document(path.read_bytes(), path.name)
            resume, _ = extract_locally(parsed["text"], parsed.get("blocks", []))
            data = resume.model_dump(mode="json", exclude_none=True)
            names += data["basicInfo"]["fullName"] not in {"", "Unknown"}
            positions += data["basicInfo"]["position"] != "Imported Resume"
            emails += bool(data["contactInfo"]["primaryEmail"])
            skills += sum(len(items) for items in data["skills"]["skills"].values()) > 0
            education += bool(data["educations"]["qualifications"])
            experience += bool(data.get("workExperience", {}).get("companies"))
            projects += bool(data.get("projects", {}).get("projects"))
            certifications += bool(data.get("certifications", {}).get("certificates"))
        except Exception:
            failures += 1

    total = len(files)
    return {
        "files_found": total,
        "files_processed": total - failures,
        "processing_failures": failures,
        "local_parser_nonempty": {
            "name": names,
            "position_nondefault": positions,
            "email": emails,
            "skills": skills,
            "education": education,
            "work_experience": experience,
            "projects": projects,
            "certifications": certifications,
        },
        "warning": (
            "Coverage counts are not extraction accuracy. Provide independently "
            "annotated expected fields to calculate precision, recall, and completeness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing sample resumes")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON summary")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")
    result = evaluate(args.directory)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
