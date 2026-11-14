from src.pdf_loader import extract_text_from_pdf

pages = extract_text_from_pdf("data/uploaded_pdfs/sample.pdf")

for page in pages[:5]:
    print(f"\n--- PAGE {page['page']} ---")
    print(page["text"][:1000])