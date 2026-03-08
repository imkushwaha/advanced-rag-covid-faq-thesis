
"""
Generation and Inference Module

This module implements the final generation stage of the RAG (Retrieval-Augmented Generation)
pipeline for the COVID-19 question-answering system. It uses a Large Language Model with
structured output to generate accurate, grounded answers based on retrieved and compressed context.

Key features:
- Factual and cautious prompting to ensure answers are grounded in provided context
- Prevention of hallucination through explicit instructions to avoid external knowledge
- Structured output using Pydantic models for consistent response formatting
- Context formatting and augmentation with chat-style messaging

Components:
- SYSTEM_PROMPT: Instructs the LLM to be factual, cautious, and context-aware
- INSTRUCTION_PROMPT: Template for formatting context and user questions
- build_augmented_prompt: Constructs chat messages with context and questions
- GenerationOutput: Pydantic model for structured response validation
- StructuredInferenceService: High-level service for generating grounded, final answers
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Dict


SYSTEM_PROMPT = """
You are a helpful, factual, and cautious AI assistant.
Your task is to answer user questions using ONLY the provided context.
If the answer cannot be found in the context, clearly say:
"I do not have enough information to answer this question."

Do not make assumptions.
Do not add external knowledge.
Be concise and accurate.
"""


INSTRUCTION_PROMPT = """
Context:
{context}

User Question:
{question}

Instructions:
- Use the context above as your primary source of information
- Do not hallucinate or fabricate facts
- If the context is insufficient, explicitly say so

Answer:
"""


def build_augmented_prompt(
    question: str,
    context_chunks: List[str],
) -> List[dict]:
    """
    Builds a structured chat-style prompt with system and user messages.

    Args:
        question (str): The user's question to be answered.
        context_chunks (List[str]): A list of context chunks to provide as sources.

    Returns:
        List[dict]: A list of message dictionaries with 'role' and 'content' keys, formatted for API calls.
    """

    context_text = "\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    user_prompt = INSTRUCTION_PROMPT.format(
        context=context_text,
        question=question,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class GenerationOutput(BaseModel):
    """
    Structured output schema for final RAG generation.
    """
    llm_response: str = Field(
        description="Final grounded answer generated using the provided context."
    )
    

class StructuredInferenceService:
    """
    Final RAG inference service with enforced structured output.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 1012,
    ):
        """
        Initializes the StructuredInferenceService with the specified LLM parameters.

        Args:
            model_name (str): The name of the LLM model to use. Defaults to "gpt-4o-mini".
            temperature (float): The temperature for response generation. Defaults to 0 for deterministic output.
            max_tokens (int): The maximum number of tokens for the response. Defaults to 1012.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ).with_structured_output(GenerationOutput)

    def generate(self, messages: List[Dict]) -> GenerationOutput:
        """
        Generates a grounded, structured response from the LLM based on the input messages.

        Args:
            messages (List[Dict]): A list of message dictionaries with 'role' and 'content' keys.

        Returns:
            GenerationOutput: A structured output object containing the generated answer.
        """
        return self.llm.invoke(messages)


# Usage Eample

# inference_llm = StructuredInferenceService()

# result = inference_llm.generate(
#     messages=build_augmented_prompt(
#         user_query,
#         compressed_context,
#     )
# )

# final_answer = result.llm_response