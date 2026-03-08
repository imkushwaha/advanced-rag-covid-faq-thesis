"""
Query Expansion Module

Query expansion (also referred to as multi-query) is a vital pre-retrieval optimisation technique used in advanced RAG systems to overcome the inherent limitations of distance-based similarity searches. Because a single query embedding may only cover a small area of the embedding space, it risks missing semantically related documents that don't sit immediately adjacent to that specific query vector.

The Concept of Query Expansion

The core idea is to use a Large Language Model (LLM) to "rewrite" the original user input into multiple different versions. By generating these different perspectives, the retrieval module can capture various facets and interpretations of the same question. This diversifies the search terms, which significantly increases the likelihood of retrieving a more comprehensive and relevant set of data points from the vector database.

Example: An initial query like "Write an article about the best types of advanced RAG methods" might be expanded into variations such as "What are the most effective advanced RAG methods, and how can they be applied?" or "Can you provide an overview of the top advanced retrieval-augmented generation techniques?".

Implementation Details

The implementation is modular and integrated into the broader retrieval workflow:

- The QueryExpander Class: This class manages the interaction with an LLM (typically GPT-4o-mini) to generate the expanded questions.
- Prompt Engineering: The system utilises a prompt template which provides a zero-shot prompt. This template instructs the LLM to generate alternative questions separated by a specific token, such as #next-question#, which the system uses to parse the generated string back into a list of query objects.
"""

from openai import OpenAI
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate



class QueryExpander:
    """
    A class for LLM-based query expansion to generate multiple query variations for improved retrieval in RAG systems.
    """

    DELIMITER = "#next-question#"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        n_variations: int = 3,
        temperature: float = 0.3,
    ):
        """
        Initializes the QueryExpander with the specified model and parameters.

        Args:
            model_name (str): The name of the LLM model to use. Defaults to "gpt-4o-mini".
            n_variations (int): The number of query variations to generate. Defaults to 3.
            temperature (float): The temperature for the LLM. Defaults to 0.3.
        """
        self.n_variations = n_variations
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

        self.prompt = PromptTemplate(
            input_variables=["query", "n", "delimiter"],
            template="""
You are an expert search query rewriter for retrieval-augmented generation systems.

Rewrite the following user query into {n} alternative versions.
Each version should reflect a different perspective or phrasing
while preserving the original intent.

Separate each rewritten query using the token:
{delimiter}

User query:
"{query}"
""",
        )

    def expand(self, query: str) -> List[str]:
        """
        Expands the given query into multiple variations using the LLM.

        Args:
            query (str): The original query to expand.

        Returns:
            List[str]: A list of queries, starting with the original, followed by generated variations.
        """
        formatted_prompt = self.prompt.format(
            query=query,
            n=self.n_variations,
            delimiter=self.DELIMITER,
        )

        response = self.llm.invoke(formatted_prompt).content

        expanded_queries = [
            q.strip()
            for q in response.split(self.DELIMITER)
            if q.strip()
        ]

        # Always include original query
        return [query] + expanded_queries



# Usage
query_expander = QueryExpansion()
expanded_queries = query_expander.run("What is the meaning of life?")

# ['1. What does it mean to have a purpose in life?',
#  '2. How can we define the significance of existence?',
#  "3. What is the philosophical interpretation of life's meaning?"]
