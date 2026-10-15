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

| Benchmark               | Definition                                                                                                                                                                                                                      | Result (POC) |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Intent Routing Accuracy | Measures how reliably the pipeline’s classification component maps an incoming user query to its correct operational category                                                                                               | 100.0%       |
| Retrieval Hit Rate      | Evaluates the vector database's retrieval engine to determine whether the exact target source document containing the required ground-truth answer was successfully retrieved within the top-K returned chunks                | 100.0%       |
| Semantic Similarity     | Evaluates the conceptual and factual alignment between the LLM's generated response and the ground-truth reference answer using dense vector embeddings                                                                         | 0.8366       |
| Lexical F1 Score        | Measures the exact string and token (word) overlap between the generated response and the ground-truth text, balancing Precision (how many generated words were correct) and Recall (how many ground-truth words were captured) | 0.4224       |

> [!NOTE]
> Generative LLMs naturally paraphrase and structure clinical responses. Evaluating via semantic similarity provides an accurate measure of model fidelity, whereas Lexical F1 serves as a diagnostic indicator for verbatim term repetition.

## CI/CD Pipeline and Automation

This project includes an automated CI/CD pipeline built with GitHub Actions (`.github/workflows/ci.yml`) that does the following on every `git push` and pull request:

* Executes Python linting, validates all XML/DITA maps and topics against structural rules, and verifies task prompt structures against `prompt-schema.json`.
* Converts the XML-based DITA taxonomy into the YAML taxonomy.
* Processes XML-based DITA topics into structured JSON files.

> [!NOTE]
> To keep the CI pipeline focused on code validation and prevent merge conflicts from generated files, the vector database ingestion and live LLM evaluation are performed locally. Run`python scripts/ingest-vector-db.py`and `python scripts/evaluate-rag.py` on your machine when updating content or benchmarking model performance.

### Recommended Branch Flow

To contribute or make updates to this repository, follow this standard branch workflow:

1. Create a Feature branch from `main` for any content updates or script modifications (e.g., `git checkout -b feature/update-dosage-content`).
2. Push your changes to your remote branch. The automated GitHub Actions pipeline will run validations and conversions.
3. Verify that all CI checks pass successfully, including XML and DITA validation.
4. Once checks pass, merge your changes into `main`. Run preprocessing and vector ingestion locally to update your local search index.
