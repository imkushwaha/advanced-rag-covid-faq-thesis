"""
Intent Classification Module

This module provides functionality for classifying user queries into predefined intent categories
for a COVID-19 question-answering system. It uses a Large Language Model (LLM) with structured output
to analyze queries and assign them to one of the following categories:

- FACTUAL_FAQ: Short, direct factual questions about definitions, symptoms, transmission, vaccines, timelines.
- PROCEDURAL_GUIDANCE: Questions asking for steps, actions, prevention, or advice.
- COMMUNITY_EXPERIENCE: Informal, opinionated, rumor-based, or social media influenced queries.
- EVIDENCE_BASED: Research-oriented, scientific, or evidence-seeking questions.
- COMPLEX_MULTI_HOP: Long, multi-part, ambiguous, or mixed-intent questions requiring synthesis.

The classification leverages a detailed system prompt that instructs the LLM to use semantic understanding
to select the single most appropriate category, ensuring accurate routing for retrieval and generation.
"""

import os
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI

api_key = "<ENTER-YOUR-OPENAI-KEY-HERE>"
os.environ["OPENAI_API_KEY"] = api_key


class IntentCategory(str, Enum):
    FACTUAL_FAQ = "FACTUAL_FAQ"
    PROCEDURAL_GUIDANCE = "PROCEDURAL_GUIDANCE"
    COMMUNITY_EXPERIENCE = "COMMUNITY_EXPERIENCE"
    EVIDENCE_BASED = "EVIDENCE_BASED"
    COMPLEX_MULTI_HOP = "COMPLEX_MULTI_HOP"


class IntentOutput(BaseModel):
    intent: IntentCategory = Field(
        description="The single most appropriate intent category for the user query"
    )


INTENT_CLASSIFIER_SYSTEM_PROMPT = """
You are an intent classification engine for a COVID-19 question-answering system.

Your task:
Given a user query, classify it into EXACTLY ONE intent category.

You must always return one category, even if the query does not clearly match any rules.
Use semantic understanding, not just keywords.

--------------------
INTENT CATEGORIES
--------------------

FACTUAL_FAQ:
- Short, direct factual questions
- Definitions, symptoms, transmission, vaccines, timelines
- Example: "What are the symptoms of COVID-19?"

PROCEDURAL_GUIDANCE:
- Questions asking for steps, actions, prevention, or advice
- Often includes "how", "what should I do", "can I", but do NOT rely only on keywords
- Example: "How can I protect myself from COVID-19?"

COMMUNITY_EXPERIENCE:
- Informal, opinionated, rumor-based, or social media influenced queries
- Mentions of people, beliefs, claims, or uncertainty
- Example: "People say masks don't work, is that true?"

EVIDENCE_BASED:
- Research-oriented, scientific, or evidence-seeking questions
- Mentions studies, data, research, clinical findings, or comparisons
- Example: "What does research say about COVID survival on surfaces?"

COMPLEX_MULTI_HOP:
- Long, multi-part, ambiguous, or mixed-intent questions
- Requires synthesising multiple facts or evidence
- Use this ONLY when no single intent clearly dominates
- Example: "How does COVID affect elderly people and what precautions should families take?"

--------------------
DECISION RULES (GUIDANCE, NOT HARD RULES)
--------------------

1. Use semantic meaning first, keywords second.
2. If multiple intents seem present:
   - Choose the MOST dominant intent.
3. If the query is long, compound, or unclear:
   - Prefer COMPLEX_MULTI_HOP.
4. If safety or scientific grounding is implied:
   - Prefer EVIDENCE_BASED over FACTUAL_FAQ.
5. Never return multiple categories.
6. Never invent new categories.
7. Never explain your reasoning.

Return ONLY the intent category in structured form.
"""



intent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

intent_classifier = intent_llm.with_structured_output(IntentOutput)


def classify_intent_llm(query: str) -> IntentCategory:
    """
    Classifies the intent of the given user query using the configured LLM.

    Args:
        query (str): The user query to classify into an intent category.

    Returns:
        IntentCategory: The single most appropriate intent category for the query.
    """
    result = intent_classifier.invoke(
        [
            {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    return result.intent


# Usage Example

queries = [
    "What are the common symptoms of COVID-19?",
    "How should I isolate if I test positive?",
    "I read online that vaccines cause infertility, is it true?",
    "According to studies, how long does COVID survive on surfaces?",
    "Can COVID affect children differently and what precautions should parents take?"
]

for q in queries:
    print(q, "→", classify_intent_llm(q))


# What are the common symptoms of COVID-19? → IntentCategory.FACTUAL_FAQ
# How should I isolate if I test positive? → IntentCategory.PROCEDURAL_GUIDANCE
# I read online that vaccines cause infertility, is it true? → IntentCategory.COMMUNITY_EXPERIENCE
# According to studies, how long does COVID survive on surfaces? → IntentCategory.EVIDENCE_BASED
# Can COVID affect children differently and what precautions should parents take? → IntentCategory.COMPLEX_MULTI_HOP