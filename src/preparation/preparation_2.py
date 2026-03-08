# Data Preparation: Dataset 2

"""
This module prepares Dataset 2 for the RAG (Retrieval-Augmented Generation) system focused on COVID-19 FAQ data.

It provides functionality to:
- Clean text by removing extra whitespace and handling missing values
- Build document strings from FAQ components (title, question, answer)
- Chunk text into smaller segments based on sentence tokenization and token limits
- Create chunks with comprehensive metadata for indexing and retrieval

Key components:
- Text cleaning using regex for whitespace normalization
- Sentence-based chunking with configurable token limits (default 300) and overlap (default 50)
- Metadata enrichment including question/answer IDs, chunk IDs, source info, and dataset labels
- Integration with NLTK for sentence tokenization and tiktoken for token counting
"""

import re
import pandas as pd
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download('punkt_tab')

enc = tiktoken.get_encoding("cl100k_base")

def clean_text(text: str) -> str:
    """
    Cleans the input text by normalizing whitespace and handling NaN values.

    Args:
        text (str): The text to clean. Can be NaN.

    Returns:
        str: The cleaned text with normalized whitespace, or empty string if input is NaN.
    """
    if pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_document(row: pd.Series) -> str:
    """
    Builds a formatted document string from FAQ components in the row.

    Args:
        row (pd.Series): A pandas Series containing optional 'title', 'question', and 'answer' keys.

    Returns:
        str: A newline-separated string with the formatted title, question, and answer.
    """
    parts = []

    if row.get("title"):
        parts.append(f"Title: {clean_text(row['title'])}")

    if row.get("question"):
        parts.append(f"Question: {clean_text(row['question'])}")

    if row.get("answer"):
        parts.append(f"Answer: {clean_text(row['answer'])}")

    return "\n".join(parts)


def chunk_text(
    text: str,
    max_tokens: int = 300,
    overlap: int = 50
):
    """
    Chunks the input text into smaller segments based on sentence boundaries and token limits.

    Args:
        text (str): The text to chunk.
        max_tokens (int, optional): Maximum number of tokens per chunk. Defaults to 300.
        overlap (int, optional): Number of sentences to overlap between chunks. Defaults to 50.

    Returns:
        list: A list of text chunks as strings.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(enc.encode(sent))

        if current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap else []
            current_tokens = len(enc.encode(" ".join(current_chunk)))

        current_chunk.append(sent)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_chunks_with_metadata(df: pd.DataFrame):
    """
    Creates text chunks with associated metadata from the input dataframe.

    Args:
        df (pd.DataFrame): A dataframe containing columns like 'document_text', 'question_id', etc.

    Returns:
        list: A list of dictionaries, each with 'text' and 'metadata' keys.
    """
    records = []

    for _, row in df.iterrows():
        chunks = chunk_text(row["document_text"])

        for i, chunk in enumerate(chunks):
            records.append({
                "text": chunk,
                "metadata": {
                    "question_id": row["question_id"],
                    "answer_id": row["answer_id"],
                    "chunk_id": i,
                    "source": row["source"],
                    "url": row["url"],
                    "answer_type": row["answer_type"],
                    "dataset": "COVID-QA-community"
                }
            })
    return records