
# Advanced Multi-Phase Retrieval-Augmented Generation for COVID-19 FAQ Question Answering

This repository contains the implementation code and experimental notebooks developed for the MSc thesis:
“Design and Evaluation of an Advanced Retrieval-Augmented Generation Workflow for COVID-19 Public Health FAQs.”
This research proposes and evaluates a multi-phase Retrieval-Augmented Generation (RAG) architecture designed to improve contextual grounding, retrieval precision, and response reliability for COVID-19 related question answering and also to perform comparative analysis of Open-Source and Closed-Source LLMs performance.

The repository is shared to support implementation transparency, reproducibility, and research review.

# Research Overview
Large Language Models (LLMs) can generate fluent responses but often suffer from factual inaccuracies when answering domain-specific questions. Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving relevant information before generating responses.

However, many traditional RAG implementations focus primarily on the generation stage while underutilising optimisation opportunities across the retrieval pipeline.

This research introduces a multi-phase RAG workflow that improves system performance through optimisation at three stages:
- Pre-Retrieval Stage
- Retrieval Stage
- Post-Retrieval Stage

The system is evaluated using both open-source and proprietary large language models.

# Key Components of the Proposed System

## Pre-Retrieval Optimisation
- Data preprocessing and transformation
- Semantic normalisation
- Query expansion
- Intent classification
- Vector embedding generation
- Multi-dataset indexing

## Retrieval Stage
- Multi-index routing
- Vector similarity search
- Context retrieval using FAISS

## Post-Retrieval Stage
- Cross-encoder re-ranking
- Prompt compression
- Context filtering before generation

## Generation Stage
- Response generation using multiple large language models to compare model behaviour under the same RAG architecture.

## Evaluation Framework
A dedicated RAG evaluation pipeline was developed to evaluate response quality using multiple metrics.
### Evaluation metrics include:
- Faithfulness
- Context Precision
- Context Recall
- Answer Relevancy
- Summarisation Score
- BLEU Score
- ROUGE-L Score
- F1 Score
- Exact Match

## Models Evaluated
The system was evaluated using five large language models.
### Proprietary Models
- GPT-4o-mini
- GPT-5-mini-2025
- GPT-5.2-2025
### pen-Source Models
- Llama-3.1-8B
- Llama-3.3-70B

Each model was executed across five independent runs, resulting in 25 total experiments.

## Repository Structure
~~~
data_preparation/
retrieval/
post_retrieval/
generation/
evaluation_pipeline/
analysis/
~~~

Each notebook corresponds to a specific stage of the RAG pipeline implemented in the thesis.

## Experimental Environment
The implementation was developed using:
- Python 
- Google Colab
- FAISS vector search
- OpenAI API
- GROQ API for Open-Source LLM
- Sentence-Transformers
- RAGAS evaluation framework
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Purpose of This Repository
This repository accompanies the MSc thesis submitted to Liverpool John Moores University. It is provided to allow reviewers and researchers to understand how the proposed advanced RAG workflow was implemented and evaluated.

The code and notebooks demonstrate the full experimental pipeline including:
- Data preparation
- Retrieval pipeline design
- LLM inference
- Automated evaluation
- Experimental analysis

## License
This repository is shared for academic and research review purposes only. Please refer to the LICENSE file for usage restrictions.
