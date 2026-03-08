
"""
Multi-Index Retriever Module

This module provides functionality for retrieving documents from multiple FAISS vector stores
based on user intent in a COVID-19 question-answering system. It supports concurrent retrieval
across expanded queries, intent-based routing to select appropriate vector stores, and
deduplication to avoid duplicate results.

Key features:
- Loading multiple FAISS vector stores from disk with OpenAI embeddings
- Intent-based routing using predefined mappings (e.g., FACTUAL_FAQ routes to specific sources)
- Concurrent retrieval using ThreadPoolExecutor for performance
- Document deduplication based on metadata ID or content hash
- Optional similarity score retrieval

The module includes predefined vector store paths for various COVID-19 datasets and intent
routing configurations to optimize retrieval relevance.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Tuple


api_key = "<ENTER-YOUR-OPENAI-KEY-HERE>"
os.environ["OPENAI_API_KEY"] = api_key


# Load Multiple FAISS Vector Stores from Google Drive

def load_faiss_vectorstores(
    vectorstore_paths: List[str],
    embedding_model: OpenAIEmbeddings,
) -> List[FAISS]:
    """
    Loads multiple FAISS vector stores from the specified local paths.

    Args:
        vectorstore_paths (List[str]): A list of file paths to the FAISS vector store directories.
        embedding_model (OpenAIEmbeddings): The embedding model used to load the vector stores.

    Returns:
        List[FAISS]: A list of loaded FAISS vector store objects.
    """
    vectorstores = []

    for path in vectorstore_paths:
        vs = FAISS.load_local(
            folder_path=path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,  # required for FAISS
        )
        vectorstores.append(vs)

    return vectorstores


# Vector Stores Path

vectorstore_paths = {
  "source_1": "/content/drive/MyDrive/MS-LJMU/Vector-Store/Store-1-3072-Kaggle-COVID-19-FAQ",

  "source_2": "/content/drive/MyDrive/MS-LJMU/Vector-Store/Store-2-3072-COVID-QA-community",

  "source_3": "/content/drive/MyDrive/MS-LJMU/Vector-Store/Store-3-3072-COUGH-FAQ-ENG",

  "source_4": "/content/drive/MyDrive/MS-LJMU/Vector-Store/Store-4-3072-COVID-QA-MASTER"

}


# Intent Routing

INTENT_ROUTING = {
    "FACTUAL_FAQ": ["source_1", "source_3"],
    "PROCEDURAL_GUIDANCE": ["source_1", "source_3"],
    "COMMUNITY_EXPERIENCE": ["source_2"],
    "EVIDENCE_BASED": ["source_4"],
    "COMPLEX_MULTI_HOP": ["source_1", "source_3", "source_4"],
}


# Multi-Index Retriever with Concurrent Retrieval

class MultiFAISSRetriever:
    """
    Concurrent retriever supporting:
    - multiple FAISS indexes
    - multiple expanded queries
    - optional similarity scores
    """

    def __init__(
        self,
        vectorstores: List[FAISS],
        top_k: int = 5,
        max_workers: int = 8,
        use_scores: bool = False,
    ):
        """
        Initializes the MultiFAISSRetriever with the given vector stores and parameters.

        Args:
            vectorstores (List[FAISS]): A list of FAISS vector store objects to search.
            top_k (int, optional): The number of top documents to retrieve per query. Defaults to 5.
            max_workers (int, optional): The maximum number of threads for concurrent retrieval. Defaults to 8.
            use_scores (bool, optional): Whether to include similarity scores in results. Defaults to False.
        """
        self.vectorstores = vectorstores
        self.top_k = top_k
        self.max_workers = max_workers
        self.use_scores = use_scores

    def _search(
        self, vs: FAISS, query: str
    ) -> List[Tuple[Document, float]] | List[Document]:
        """
        Performs similarity search on a single vector store for the given query.

        Args:
            vs (FAISS): The FAISS vector store to search.
            query (str): The query string to search for.

        Returns:
            List[Tuple[Document, float]] | List[Document]: A list of documents, with scores if use_scores is True.
        """
        if self.use_scores:
            return vs.similarity_search_with_score(query, k=self.top_k)
        return vs.similarity_search(query, k=self.top_k)

    def retrieve(
        self, queries: List[str]
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Executes concurrent retrieval across all vector stores and queries, then deduplicates results.

        Args:
            queries (List[str]): A list of query strings to search for.

        Returns:
            List[Document] | List[Tuple[Document, float]]: A list of unique documents, with scores if use_scores is True.
        """

        seen_doc_ids = set()
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._search, vs, q)
                for vs in self.vectorstores
                for q in queries
            ]

            for future in as_completed(futures):
                retrieved = future.result()

                for item in retrieved:
                    if self.use_scores:
                        doc, score = item
                    else:
                        doc = item
                        score = None

                    # Robust deduplication key
                    doc_id = (
                        doc.metadata.get("id")
                        if doc.metadata and "id" in doc.metadata
                        else hash(doc.page_content)
                    )

                    if doc_id in seen_doc_ids:
                        continue

                    seen_doc_ids.add(doc_id)

                    if self.use_scores:
                        results.append((doc, score))
                    else:
                        results.append(doc)

        return results



# Usage Example

user_query = "What are the common symptoms of COVID-19?"

# Indentify the intent and load respective vector stores
# intent = classify_intent_llm(user_query) Load it from intent_classification.py

# sources = INTENT_ROUTING[intent]
# stores_paths = [vectorstore_paths[source] for source in sources]

# indexes = load_faiss_vectorstores(
#     vectorstore_paths=stores_paths,
#     embedding_model=embeddings,
# )


# # Expand query
# expander = QueryExpander(n_variations=3) # Load it from query_expansion.py
# expanded_queries = expander.expand(user_query)

# Retrieve documents
retriever = MultiFAISSRetriever(
    vectorstores=indexes,
    top_k=5,
    use_scores=True
)

retrieved_docs = retriever.retrieve(expanded_queries)

# [(Document(id='9f3910b0-3166-426e-9c7a-a52717843b39', metadata={'dataset': 'COVID19_FAQ', 'original_question': '2. What are the symptoms of COVID-19?', 'record_id': 1, 'chunk_id': 0, 'chunk_type': 'full', 'source': 'Kaggle COVID-19 FAQ', 'embedding_tokens': 166}, page_content="Question: 2. What are the symptoms of COVID-19?\n\nAnswer:\nThe most common symptoms of COVID-19 are fever, tiredness, and dry cough. Some patients   may have aches and pains, nasal congestion, runny nose, sore throat or diarrhea. These   symptoms are usually mild and begin gradually. Some people become infected but dont   develop any symptoms and don't feel unwell. Most people (about 80%) recover from the   disease without needing special treatment. Around 1 out of every 6 people who gets COVID-19   becomes seriously ill and develops difficulty breathing. Older people, and those with underlying   medical problems like high blood pressure, heart problems or diabetes, are more likely to   develop serious illness. People with fever, cough and difficulty breathing should seek medical   attention."),
#   np.float32(0.8246421)),
#  (Document(id='a4616ee8-ce94-4359-9eab-b7c72d0a259a', metadata={'dataset': 'COVID19_FAQ', 'original_question': '3. How do I know if it is COVID-19 or just the common flu?', 'record_id': 2, 'chunk_id': 0, 'chunk_type': 'full', 'source': 'Kaggle COVID-19 FAQ', 'embedding_tokens': 146}, page_content='Question: 3. How do I know if it is COVID-19 or just the common flu?\n\nAnswer:\nA COVID-19 infection has the same signs and symptoms as the common cold and you can only   differentiate them through laboratory testing to determine the virus type. If you have fever, cough   and difficulty breathing, you should seek medical attention and immediately isolate yourself from   others. Call your local UN clinic/medical facility to inform them of your condition and relevant   travel/exposure history. If you had been identified as a close contact of a case by the local   Ministry of Health or WHO, please also indicate this. From here you will be advised if a medical   assessment is necessary and how to get tested.'),
#   np.float32(0.99232185))]