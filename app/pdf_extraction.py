import pymupdf

def extract_text(file_path: str) -> str:
    """Read PDF and extract all text."""
    doc = pymupdf.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def save_file(file_contents: bytes, file_path: str) -> None:
    """Save uploaded file to a temporary location."""
    with open(file_path, "wb") as f:
        f.write(file_contents)
