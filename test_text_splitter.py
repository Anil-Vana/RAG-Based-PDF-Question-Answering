from src.pdf_loader import extract_text_from_pdf
from src.text_splitter import split_text

pdf_path = "data/uploaded_pdfs/sample.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = split_text(pages)

print("Total chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0]["text"])
print("\nFirst chunk page:", chunks[0]["page"])