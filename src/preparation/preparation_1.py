
# Data Preparation: Dataset 1

"""
This module prepares Dataset 1 for the RAG (Retrieval-Augmented Generation) system focused on COVID-19 FAQ data.

It provides functionality to:
- Join questions and answers in a structured, LLM-friendly format
- Perform adaptive chunking based on word counts for FAQ data
- Prepare embedding records with metadata for vector database ingestion

Key components:
- Word-based chunking with configurable maximum words (default 225, approximately 300 tokens)
- Adaptive strategy: single chunk for short Q+A, or chunked answers with repeated questions
- Metadata enrichment including dataset name, chunk type, and source information
- Support for Kaggle COVID-19 FAQ dataset
"""

import pandas as pd
from typing import List, Dict

def join_question_answer(question: str, answer: str) -> str:
    """
    Joins the question and answer into a structured, LLM-friendly format.

    Args:
        question (str): The question text.
        answer (str): The answer text.

    Returns:
        str: The formatted string with stripped question and answer.
    """
    return f"Question: {question.strip()}\n\nAnswer:\n{answer.strip()}"


def word_count(text: str) -> int:
    """
    Counts the number of words in the given text by splitting on whitespace.

    Args:
        text (str): The text to count words in.

    Returns:
        int: The number of words in the text.
    """
    return len(text.split())

def adaptive_chunk_qa(
    question: str,
    answer: str,
    max_words: int = 225 # 300 Tokens
) -> List[str]:
    """
    Performs adaptive chunking for FAQ-style data based on word counts.

    If the combined question and answer fits within max_words, returns a single chunk.
    Otherwise, chunks the answer only and repeats the question for each chunk.

    Args:
        question (str): The question text.
        answer (str): The answer text.
        max_words (int, optional): Maximum words per chunk. Defaults to 225.

    Returns:
        List[str]: A list of chunked text strings.
    """
    combined = join_question_answer(question, answer)

    if word_count(combined) <= max_words:
        return [combined]

    # Chunk answer only
    answer_words = answer.split()
    chunks = []

    for i in range(0, len(answer_words), max_words):
        chunk_answer = " ".join(answer_words[i:i + max_words])
        chunk_text = f"Question: {question.strip()}\n\nAnswer:\n{chunk_answer}"
        chunks.append(chunk_text)

    return chunks


def prepare_embedding_records(
    df: pd.DataFrame,
    dataset_name: str
) -> List[Dict]:
    """
    Prepares final records with text chunks and metadata for vector database ingestion.

    Args:
        df (pd.DataFrame): The dataframe containing 'questions' and 'answers' columns.
        dataset_name (str): The name of the dataset for metadata.

    Returns:
        List[Dict]: A list of dictionaries, each with 'text' and 'metadata' keys.
    """
    records = []

    for idx, row in df.iterrows():
        chunks = adaptive_chunk_qa(
            question=row["questions"],
            answer=row["answers"]
        )

        for chunk_id, chunk_text in enumerate(chunks):
            records.append({
                "text": chunk_text,
                "metadata": {
                    "dataset": dataset_name,
                    "original_question": row["questions"],
                    "record_id": idx,
                    "chunk_id": chunk_id,
                    "chunk_type": "full" if len(chunks) == 1 else "partial",
                    "source": "Kaggle COVID-19 FAQ"
                }
            })

    return records
