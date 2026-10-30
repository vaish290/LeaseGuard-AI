
import fitz  # PyMuPDF


def extract_pdf_text(uploaded_file):
    pages = []

    try:
        pdf_bytes = uploaded_file.getvalue()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "page": page_number,
                        "text": text
                    })

        if not pages:
            raise ValueError(
                "No readable text found. The PDF may be scanned."
            )

        return pages

    except Exception as error:
        raise ValueError(
            f"Unable to process PDF: {error}"
        ) from error
