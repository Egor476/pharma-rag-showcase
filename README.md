# RAG Showcase Project for Pharma Content

This is an end-to-end retrieval-augmented generation (RAG) showcase project for pharmaceutical content.

This showcase project contains a RAG pipeline that:

1. Indexes structured DITA XML topics into a vector storage layer.
2. Routes queries based on intent and target audience personas.
3. Enforces strict regulatory guardrails using local LLM inference.

> [!NOTE]
> This showcase project is only a proof of concept (POC) designed to demonstrate a basic RAG pipeline.

> [!WARNING]
> All the pharmaceutical information used for this showcase project and stored in this repository, including clinical overview, adverse reactions, dosage administration, contraindications, and precautions, is entirely fictional and created solely for demonstration purposes. It does not represent real-world clinical guidance or real drug products.

## Technical Stack

This showcase project is based on the following technologies:

* Structured content: DITA XML (DITA OT), YAML, JSON
* Vector store and indexing: ChromaDB, all-MiniLM-L6-v2
* Orchestration and LLM framework: LangChain, Ollama (llama3)

## Benchmarks and Performance

The following table includes the benchmarks used to evaluate the RAG pipeline for this showcase project:

| Benchmark                | Definition                                                                                                                                                                                                                      | Result (POC) |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Intent Routing Accuracy  | Measures how reliably the pipeline’s classification component maps an incoming user query to its correct operational category                                                                                               | 100.0%       |
| Retrieval Hit Rate       | Evaluates the vector database's retrieval engine to determine whether the exact target source document containing the required ground-truth answer was successfully retrieved within the top-K returned chunks                | 100.0%       |
| Mean Semantic Similarity | Evaluates the conceptual and factual alignment between the LLM's generated response and the ground-truth reference answer using dense vector embeddings                                                                         | 0.7958       |
| Mean Lexical F1 Score    | Measures the exact string and token (word) overlap between the generated response and the ground-truth text, balancing Precision (how many generated words were correct) and Recall (how many ground-truth words were captured) | 0.3453       |

> [!NOTE]
> Generative LLMs naturally paraphrase and structure clinical responses. Evaluating via semantic similarity provides an accurate measure of model fidelity, whereas Lexical F1 serves as a diagnostic indicator for verbatim term repetition.

## Next Steps

The next steps will be establishing a unified CI/CD pipeline that will automate the following upon `git push`:

* XML-to-YAML taxonomy conversion
* XML-to-JSON topic conversion
* ChromaDB vector embeddings
* RAG evaluation

A quality gate will be implemented to prevent pushing changes that have led to lower benchmark results.
