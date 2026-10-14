import pymupdf

from app.services.file_extractor import extract_document


def test_pdf_with_blank_page_keeps_text_without_unnecessary_ocr():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Jane Doe resume")
    document.new_page()  # intentionally blank page
    data = document.tobytes()
    document.close()

    parsed = extract_document(data, "resume.pdf")
    assert "Jane Doe resume" in parsed["text"]
    assert any(block["text"] == "Jane Doe resume" for block in parsed["blocks"])


def test_only_pdf_and_docx_are_supported():
    import pytest
    from app.services.file_extractor import UnsupportedFileTypeError

    with pytest.raises(UnsupportedFileTypeError, match="Upload a PDF or DOCX"):
        extract_document(b"not an image", "resume.png")
    with pytest.raises(UnsupportedFileTypeError, match="Upload a PDF or DOCX"):
        extract_document(b"plain text", "resume.txt")
