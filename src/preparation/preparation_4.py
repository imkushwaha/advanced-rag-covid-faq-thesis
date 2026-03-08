# Data Preparation: Dataset 4

"""
This module prepares Dataset 4 for the RAG (Retrieval-Augmented Generation) system focused on COVID-19 FAQ data.

It provides functionality to:
- Chunk FAQ answers into smaller text segments using RecursiveCharacterTextSplitter
- Format chunks with associated questions
- Count tokens in text using tiktoken for embedding purposes
- Add token count metadata to documents

Key components:
- Text splitter configured with 300 character chunks and 50 character overlap
- Token counting using the specified embedding model's encoding
- Metadata enrichment for downstream processing
"""

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

def chunk_faq(row):
    """
    Chunks the FAQ answer into smaller text segments and formats each chunk with the associated question.

    Args:
        row (dict): A dictionary containing 'question' and 'answer' keys representing an FAQ entry.

    Returns:
        list: A list of tuples, each containing the formatted text (str) and the chunk index (int).
    """
    chunks = splitter.split_text(row["answer"])
    documents = []

    for i, chunk in enumerate(chunks):
        text = f"""Question: {row['question']}

Answer (Part {i+1}): {chunk}"""
        documents.append((text, i))

    return documents


encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the given text using tiktoken encoding.

    Args:
        text (str): The input text to count tokens for.

    Returns:
        int: The number of tokens in the text.
    """
    return len(encoding.encode(text))


def add_token_metadata(documents):
    """
    Adds token count metadata to each document in the list.

    Args:
        documents (list): A list of Document objects to update with token metadata.

    Returns:
        list: The same list of documents with 'embedding_tokens' added to their metadata.
    """
    for doc in documents:
        doc.metadata["embedding_tokens"] = count_tokens(doc.page_content)
    return documents