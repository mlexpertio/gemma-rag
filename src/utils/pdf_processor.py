import pymupdf4llm
import os


def convert_pdf_to_md(pdf_path: str, output_dir: str) -> str:
    """
    Converts a PDF file to Markdown format using pymuprypt4llm.
    Returns the path to the created markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.basename(pdf_path)
    md_file_name = os.path.splitext(file_name)[0] + ".md"
    md_path = os.path.join(output_dir, md_file_name)

    # Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(str(md_text))

    return md_path
