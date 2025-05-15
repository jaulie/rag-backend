import pymupdf

def extract_pages(file_path: str) -> str:
    """Read PDF and extract all text."""
    doc = pymupdf.open(file_path)
    pages = [page.get_text().strip() for page in doc]
    doc.close()
    return [p for p in pages if p]

def save_file(file_contents: bytes, file_path: str) -> None:
    """Save uploaded file to a temporary location."""
    with open(file_path, "wb") as f:
        f.write(file_contents)
