from document_loader import load_pdf


def chunk_text(text, chunk_size=1000, overlap=150):
    """
    Create paragraph-aware chunks with overlap between chunks.
    """

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If paragraph fits in current chunk
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"

        else:
            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Create overlap from the end of previous chunk
            overlap_text = current_chunk[-overlap:]

            # Start next chunk with overlap
            current_chunk = overlap_text + "\n\n" + paragraph + "\n\n"

    # Save final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_document_chunks(pdf_path):
    pages = load_pdf(pdf_path)

    all_chunks = []

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append({
                "page_number": page_number,
                "text": chunk
            })

    return all_chunks