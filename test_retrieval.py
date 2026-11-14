
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import (
    keyword_search,
    reciprocal_rank_fusion
)


# ==========================================
# LOAD EXISTING VECTOR STORE
# ==========================================

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    embedding_model,
    "data/vectorstore"
)


# ==========================================
# EVALUATION QUESTIONS
# ==========================================

evaluation_data = [
    {
        "question": "Which algorithm performed best and what was its F1-score?",
        "relevant_pages": [5]
    },
    {
        "question": "What was the ROC-AUC score of Random Forest?",
        "relevant_pages": [5]
    },
    {
        "question": "What was the F1-score of the Decision Tree classifier?",
        "relevant_pages": [5]
    },
    {
        "question": "How many clusters were generated using K-Means?",
        "relevant_pages": [5]
    },
    {
        "question": "What was the silhouette score of K-Means?",
        "relevant_pages": [5]
    }
]


# ==========================================
# RETRIEVAL EVALUATION
# ==========================================

methods = {
    "Semantic": [],
    "Keyword": [],
    "Hybrid": []
}


for item in evaluation_data:

    question = item["question"]
    relevant_pages = item["relevant_pages"]

    # Semantic search
    semantic_results = (
        vector_store.similarity_search_with_score(
            question,
            k=10
        )
    )

    semantic_documents = [
        {"document": document}
        for document, score in semantic_results
    ]

    # Keyword search
    keyword_results = keyword_search(
        vector_store,
        question,
        k=10
    )

    # Hybrid search
    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results
    )

    retrieval_results = {
        "Semantic": semantic_documents,
        "Keyword": keyword_results,
        "Hybrid": hybrid_results
    }

    for method, results in retrieval_results.items():

        retrieved_pages = [
            result["document"].metadata.get("page")
            for result in results
        ]

        hit_5 = int(
            any(
                page in relevant_pages
                for page in retrieved_pages[:5]
            )
        )

        hit_10 = int(
            any(
                page in relevant_pages
                for page in retrieved_pages[:10]
            )
        )

        reciprocal_rank = 0

        for rank, page in enumerate(
            retrieved_pages,
            start=1
        ):

            if page in relevant_pages:
                reciprocal_rank = 1 / rank
                break

        methods[method].append(
            (hit_5, hit_10, reciprocal_rank)
        )


# ==========================================
# DISPLAY EVALUATION RESULTS
# ==========================================

print("\nRETRIEVAL EVALUATION RESULTS")
print("=" * 65)

for method, results in methods.items():

    hit_rate_5 = sum(
        result[0] for result in results
    ) / len(results)

    hit_rate_10 = sum(
        result[1] for result in results
    ) / len(results)

    mrr = sum(
        result[2] for result in results
    ) / len(results)

    print(f"\n{method} Search")

    print(f"Hit Rate@5  : {hit_rate_5:.4f}")
    print(f"Hit Rate@10 : {hit_rate_10:.4f}")
    print(f"MRR         : {mrr:.4f}")

print("\nEvaluation completed.")
