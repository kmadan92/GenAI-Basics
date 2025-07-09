# fanout_rrf.py

from typing import List, Tuple
from langchain_core.vectorstores import VectorStoreRetriever


def reciprocal_rank_fusion(results: List[List[Tuple[str, float]]], k: int = 3) -> List[str]:
    """
    Combine results from multiple retrievers or query variations using Reciprocal Rank Fusion (RRF).

    Args:
        results (List[List[Tuple[str, float]]]): A list of result lists. Each inner list contains tuples of 
            (page_content, similarity_score). Lower scores typically indicate higher relevance.
        k (int): Number of final top results to return after fusion.

    Returns:
        List[str]: List of top k page contents after applying reciprocal rank fusion.
    """
    scores = {}
    for retriever_results in results:
        for rank, (content, _) in enumerate(retriever_results):
            # RRF score: 1 / (rank + 1)
            scores[content] = scores.get(content, 0) + 1 / (rank + 1)

    # Sort by cumulative RRF score (descending)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked[:k]]


def fanout_rrf_query_multiqueries(
    queries: List[str],
    retrievers: List[VectorStoreRetriever],
    top_k_per_ret: int = 5,
    final_k: int = 3
) -> List[str]:
    """
    Perform fan-out retrieval for multiple query variants across multiple retrievers,
    and combine the results using Reciprocal Rank Fusion (RRF).

    Args:
        queries (List[str]): List of query variations including the original.
        retrievers (List[VectorStoreRetriever]): LangChain retrievers (typically one per document set or collection).
        top_k_per_ret (int): Number of top documents to retrieve per retriever for each query.
        final_k (int): Final number of top results to return after fusion.

    Returns:
        List[str]: Final top-k ranked chunk texts after fan-out + RRF.
    """
    all_results = []

    for query in queries:
        for retriever in retrievers:
            # similarity_search_with_score returns List[Tuple[Document, float]]
            docs_with_scores = retriever.similarity_search_with_score(query=query, k=top_k_per_ret)
            # Convert to (page_content, score) format
            chunk_score_pairs = [
                (doc.page_content.strip(), score) for doc, score in docs_with_scores
            ]
            all_results.append(chunk_score_pairs)

    return reciprocal_rank_fusion(all_results, k=final_k)
