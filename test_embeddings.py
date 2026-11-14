from src.pdf_loader import extract_text_from_pdf
from src.text_splitter import split_text
from src.embeddings import get_embedding_model

pages = extract_text_from_pdf("data/uploaded_pdfs/sample.pdf")
chunks = split_text(pages)

embedding_model = get_embedding_model()

text = chunks[0]["text"]
vector = embedding_model.embed_query(text)

print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])