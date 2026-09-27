try:
    import fitz  # type: ignore[import-not-found]
except ImportError:
    fitz = None


def load_pdf(file_path):
    document = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    file_path = "data/documents/Tatamotors.pdf"
    pages = load_pdf(file_path)

    print(f"Number of pages: {len(pages)}")

    print("\nFirst page:\n")
    print(pages[0]["text"])