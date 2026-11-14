from src.rag_pipeline import initialize_rag

vector_store = initialize_rag()

query = "What is the purpose of the employee handbook?"

results = vector_store.similarity_search(
    query,
    k=3
)

for result in results:
    print("\nPage:", result.metadata["page"])
    print("Content:", result.page_content)