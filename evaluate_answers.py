
import csv
import json

from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import get_llm, answer_question


evaluation_data = [
    {
        "question": "Which algorithm performed best and what was its F1-score?",
        "expected_answer": "Random Forest performed best with an F1-score of 0.82.",
        "expected_page": 5,
    },
    {
        "question": "What was the ROC-AUC score of Random Forest?",
        "expected_answer": "Random Forest achieved a ROC-AUC score of 0.89.",
        "expected_page": 5,
    },
    {
        "question": "What was the F1-score of Decision Tree?",
        "expected_answer": "Decision Tree achieved an F1-score of 0.74.",
        "expected_page": 5,
    },
    {
        "question": "How many clusters were generated using K-Means?",
        "expected_answer": "K-Means generated six clusters.",
        "expected_page": 5,
    },
    {
        "question": "What was the silhouette score of K-Means?",
        "expected_answer": "K-Means achieved a silhouette score of 0.53.",
        "expected_page": 5,
    },
]


def get_document(source):
    if isinstance(source, dict):
        return source.get("document")
    return source


def format_sources(sources):
    context = []

    for source in sources:
        document = get_document(source)

        if document is not None:
            page = document.metadata.get("page", "Unknown")
            context.append(
                f"Page {page}:\n{document.page_content}"
            )

    return "\n\n".join(context)


def judge_answer(llm, question, expected_answer, generated_answer, context):
    prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Retrieved Context:
{context}

Evaluate:

1. Correctness: Is the answer factually correct?
2. Faithfulness: Is the answer supported by retrieved context?
3. Relevance: Does it answer the question directly?

Score each from 1 to 5.

Return ONLY valid JSON:
{{
    "correctness_score": 1,
    "correctness_reason": "Explanation",
    "faithfulness_score": 1,
    "faithfulness_reason": "Explanation",
    "relevance_score": 1,
    "relevance_reason": "Explanation"
}}
"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0].strip()

    evaluation = json.loads(content)

    for key in [
        "correctness_score",
        "faithfulness_score",
        "relevance_score",
    ]:
        evaluation[key] = int(evaluation[key])

        if not 1 <= evaluation[key] <= 5:
            raise ValueError(f"Invalid score: {key}")

    return evaluation


def main():
    print("\nRAG ANSWER EVALUATION")
    print("=" * 70)

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        embedding_model,
        "data/vectorstore",
    )

    llm = get_llm()

    results = []

    for index, item in enumerate(evaluation_data, start=1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_page = item["expected_page"]

        print(f"\nQuestion {index}: {question}")

        generated_answer, sources = answer_question(
            vector_store,
            llm,
            question,
        )

        context = format_sources(sources)

        source_pages = []

        for source in sources:
            document = get_document(source)

            if document is not None:
                page = document.metadata.get("page")

                if page is not None and page not in source_pages:
                    source_pages.append(page)

        expected_page_found = expected_page in source_pages

        try:
            evaluation = judge_answer(
                llm,
                question,
                expected_answer,
                generated_answer,
                context,
            )

            correctness_score = evaluation["correctness_score"]
            correctness_reason = evaluation["correctness_reason"]

            faithfulness_score = evaluation["faithfulness_score"]
            faithfulness_reason = evaluation["faithfulness_reason"]

            relevance_score = evaluation["relevance_score"]
            relevance_reason = evaluation["relevance_reason"]

            print(
                f"Correctness: {correctness_score}/5 "
                f"- {correctness_reason}"
            )

            print(
                f"Faithfulness: {faithfulness_score}/5 "
                f"- {faithfulness_reason}"
            )

            print(
                f"Relevance: {relevance_score}/5 "
                f"- {relevance_reason}"
            )

        except Exception as error:
            print(f"Evaluation failed: {error}")

            correctness_score = ""
            correctness_reason = str(error)

            faithfulness_score = ""
            faithfulness_reason = ""

            relevance_score = ""
            relevance_reason = ""

        print(f"Expected Answer: {expected_answer}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Expected Page: {expected_page}")
        print(f"Retrieved Pages: {source_pages}")

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "expected_page": expected_page,
            "retrieved_pages": ", ".join(map(str, source_pages)),
            "expected_page_found": expected_page_found,
            "correctness_score": correctness_score,
            "correctness_reason": correctness_reason,
            "faithfulness_score": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "relevance_score": relevance_score,
            "relevance_reason": relevance_reason,
        })

    output_file = "answer_evaluation_results.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nAVERAGE EVALUATION SCORES")

    for metric in [
        "correctness_score",
        "faithfulness_score",
        "relevance_score",
    ]:
        scores = [
            row[metric]
            for row in results
            if isinstance(row[metric], int)
        ]

        if scores:
            average = sum(scores) / len(scores)
            print(f"{metric}: {average:.2f}/5")
        else:
            print(f"{metric}: No scores available")

    print("\nAnswer evaluation completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()


