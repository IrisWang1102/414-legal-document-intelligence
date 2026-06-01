from pypdf import PdfReader


def load_document(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return "\n".join(text)

    raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")