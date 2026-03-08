"""
Post-Retrieval Reranker Module

This module implements reranking using cross-encoder models to optimize retrieved documents
in a RAG system. Reranking addresses the limitations of bi-encoder similarity searches, which
may retrieve semantically close but contextually misaligned documents.

The Problem: Initial vector searches rely on embedding distance, potentially missing nuanced
query-document relationships.

The Solution: Cross-encoder models score each retrieved chunk against the original query,
identifying complex and contextual alignments.

The Workflow: Retrieve a broader pool of candidates (e.g., N×K chunks with query expansion),
then rerank and select the top K results for the final generation.

The CrossEncoderReranker class uses sentence-transformers' CrossEncoder for efficient,
batch-processed reranking with configurable parameters.
"""


from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List, Tuple


class CrossEncoderReranker:
    """
    Cross-encoder reranker for post-retrieval optimisation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        batch_size: int = 16,
    ):
        """
        Initializes the CrossEncoderReranker with the specified model and parameters.

        Args:
            model_name (str): The name of the cross-encoder model to use. Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
            top_k (int): The number of top documents to return after reranking. Defaults to 5.
            batch_size (int): The batch size for prediction to optimize performance. Defaults to 16.
        """
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        """
        Reranks the retrieved documents using cross-encoder scores against the query.

        Args:
            query (str): The original query string.
            documents (List[Document]): A list of documents to rerank.

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing documents and their scores, sorted by score descending, limited to top_k.
        """

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        reranked = list(zip(documents, scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[: self.top_k]


# usage Example

# rerank_top_k = 5
# reranker = CrossEncoderReranker(top_k=rerank_top_k)
# reranked_docs_with_scores = reranker.rerank(
#         query=user_query,
#         documents=retrieved_docs,
#     )



