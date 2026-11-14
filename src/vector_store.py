
import chromadb
from langchain_chroma import Chroma


COLLECTION_NAME = "general_pdf_rag"


def create_vector_store(embedding_model, persist_directory):
    client = chromadb.PersistentClient(
        path=persist_directory
    )

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )
