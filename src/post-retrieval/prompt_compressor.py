"""
Prompt Compression Module

This module implements prompt compression (context refinement) to optimize retrieved documents
for generation in RAG systems. It addresses the issue of verbose or redundant chunks that can
increase costs, cause attention bias, or lead to hallucinations.

Why Prompt Compression?

Even after reranking, retrieved chunks may contain irrelevant details or redundancy.
The goal is to preserve factual meaning while removing noise, resulting in concise,
medically accurate outputs.

The PromptCompressor class uses an LLM to compress individual document chunks based on
the user query, ensuring all important facts are retained without adding new information.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List


class PromptCompressor:
    """
    Compresses retrieved documents while preserving factual content.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        """
        Initializes the PromptCompressor with the specified LLM and parameters.

        Args:
            model_name (str): The name of the LLM model to use. Defaults to "gpt-4o-mini".
            temperature (float): The temperature for the LLM. Defaults to 0.0 for deterministic output.
            max_tokens (int): The maximum tokens for the compressed output. Defaults to 256.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
You are a system that compresses retrieved context for a question-answering system.

Given the user query and the retrieved text:
- Remove irrelevant or redundant information
- Preserve all medically important facts
- Do NOT add new information
- Keep the output concise and factual

User query:
{query}

Retrieved context:
{context}

Compressed context:
""",
        )

        self.max_tokens = max_tokens

    def compress(
        self,
        query: str,
        documents: List[Document],
    ) -> List[str]:
        """
        Compresses the retrieved documents based on the query, preserving factual content.

        Args:
            query (str): The user query to guide compression.
            documents (List[Document]): A list of documents to compress.

        Returns:
            List[str]: A list of compressed context strings, one for each document.
        """
        compressed_chunks = []

        for doc in documents:
            formatted_prompt = self.prompt.format(
                query=query,
                context=doc.page_content,
            )

            response = self.llm.invoke(
                formatted_prompt,
                max_tokens=self.max_tokens,
            )

            compressed_chunks.append(response.content.strip())

        return compressed_chunks


# Usaage Example

# compressor = PromptCompressor()
# compressed_context = compressor.compress(
#     query=user_query,
#     documents=reranked_docs
#     )