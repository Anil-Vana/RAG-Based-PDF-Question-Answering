
from src.rag_pipeline import (
    initialize_rag,
    get_llm,
    answer_question
)

pdf_path = "data/uploaded_pdfs/phase 1 (6).pdf"

vector_store = initialize_rag(pdf_path)
llm = get_llm()

question = "Which algorithm performed best and what was its F1-score?"

answer, sources = answer_question(
    vector_store,
    llm,
    question
)

print("\nANSWER:\n")
print(answer)

print("\nSOURCES:\n")

for source in sources:
    document = source["document"]
    print("Page:", document.metadata.get("page"))
    print(document.page_content)
    print("-" * 50)

