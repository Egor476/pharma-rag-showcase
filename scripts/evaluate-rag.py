import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Import RAG pipeline from local script
import importlib.util
from pathlib import Path

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST_DATASET_PATH = PROJECT_ROOT / "data" / "test-dataset.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation-results.json"

# Locate rag-pipeline.py in the same scripts/ directory
_pipeline_path = Path(__file__).resolve().parent / "rag-pipeline.py"
_spec = importlib.util.spec_from_file_location("rag_pipeline_module", _pipeline_path)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec from path: {_pipeline_path}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Extract class
PharmaRAGPipeline = _module.PharmaRAGPipeline

print("[Eval Init] Loading SentenceTransformer for semantic evaluation...")
eval_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_similarity(generated_text: str, ground_truth_text: str) -> float:
    """Computes semantic similarity score using sentence embeddings."""
    if not generated_text or not ground_truth_text:
        return 0.0
    
    emb_gen = eval_embedder.encode(generated_text, convert_to_tensor=True)
    emb_gt = eval_embedder.encode(ground_truth_text, convert_to_tensor=True)
    
    similarity = util.cos_sim(emb_gen, emb_gt).item()
    return round(float(similarity), 4)

def compute_token_overlap_metrics(generated_text: str, ground_truth_text: str) -> Dict[str, float]:
    """
    Computes exact keyword overlap and Token-level Precision/Recall/F1 
    between generated answer and ground truth text (Lexical Similarity Benchmark).
    """
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    gen_tokens = tokenize(generated_text)
    gt_tokens = tokenize(ground_truth_text)

    if not gen_tokens or not gt_tokens:
        return {"lexical_precision": 0.0, "lexical_recall": 0.0, "lexical_f1": 0.0}

    common_tokens = set(gen_tokens) & set(gt_tokens)
    overlap_count = sum(min(gen_tokens.count(t), gt_tokens.count(t)) for t in common_tokens)

    precision = overlap_count / len(gen_tokens)
    recall = overlap_count / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "lexical_precision": round(precision, 4),
        "lexical_recall": round(recall, 4),
        "lexical_f1": round(f1, 4),
    }


def run_evaluation():
    print("==================================================")
    print(" PHARMA RAG PIPELINE EVALUATION BENCHMARK ")
    print("==================================================")

    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset file not found at: {TEST_DATASET_PATH}")

    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)

    test_cases = dataset_data.get("test_cases", [])
    print(f"Loaded {len(test_cases)} test cases from {TEST_DATASET_PATH.name}.\n")

    pipeline = PharmaRAGPipeline()
    results_summary = []

    total_cases = len(test_cases)
    intent_matches = 0
    source_hits = 0
    total_f1 = 0.0
    total_semantic_sim = 0.0

    for idx, tc in enumerate(test_cases, 1):
        tc_id = tc.get("id")
        query_text = tc.get("query")
        expected_intent = tc.get("intent")
        expected_sources = tc.get("expected_sources", [])
        ground_truth_ans = tc.get("ground_truth_answer", "")

        print(f"[{idx}/{total_cases}] Evaluating Test Case: {tc_id}")
        print(f"  Query: \"{query_text}\"")

        # Execute Pipeline Query
        response_payload = pipeline.query(
            query_text=query_text,
            product=tc.get("product"),
            category=tc.get("category"),
            target_audience=tc.get("target_audience", "HCP"),
            intent=expected_intent,
        )

        detected_intent = response_payload.get("intent_detected")
        generated_response = response_payload.get("response", "")
        retrieved_sources = [src.get("source_file") for src in response_payload.get("retrieved_sources", [])]

        # Metric 1: Intent Routing Accuracy
        intent_correct = (detected_intent == expected_intent)
        if intent_correct:
            intent_matches += 1

        # Metric 2: Retrieval Recall (Hit@K)
        if not expected_sources:
            # Fallback test cases expect empty sources
            hit_at_k = 1.0 if len(retrieved_sources) == 0 else 0.0
        else:
            hit_at_k = 1.0 if any(exp in retrieved_sources for exp in expected_sources) else 0.0
        
        if hit_at_k == 1.0:
            source_hits += 1

        # Metric 3: Response Lexical Similarity & Fidelity
        lexical_metrics = compute_token_overlap_metrics(generated_response, ground_truth_ans)
        semantic_sim = compute_semantic_similarity(generated_response, ground_truth_ans)

        total_f1 += lexical_metrics["lexical_f1"]
        total_semantic_sim += semantic_sim

        case_result = {
            "test_id": tc_id,
            "query": query_text,
            "expected_intent": expected_intent,
            "detected_intent": detected_intent,
            "intent_match": intent_correct,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "hit_at_k": hit_at_k,
            "ground_truth_answer": ground_truth_ans,
            "generated_response": generated_response,
            "metrics": {**lexical_metrics, "semantic_similarity": semantic_sim},
        }
        results_summary.append(case_result)

        print(f"  -> Intent Correct: {intent_correct} | Hit@K: {hit_at_k} | Lexical F1: {lexical_metrics['lexical_f1']}| Semantic Sim: {semantic_sim}\n")

    # Aggregate Benchmark Metrics
    overall_intent_accuracy = round(intent_matches / total_cases, 4)
    overall_hit_rate = round(source_hits / total_cases, 4)
    overall_mean_f1 = round(total_f1 / total_cases, 4)
    overall_mean_semantic = round(total_semantic_sim / total_cases, 4)

    benchmark_report = {
        "summary": {
            "total_test_cases": total_cases,
            "intent_routing_accuracy": overall_intent_accuracy,
            "retrieval_hit_rate_at_k": overall_hit_rate,
            "mean_lexical_f1_score": overall_mean_f1,
            "mean_semantic_similarity": overall_mean_semantic
        },
        "detailed_results": results_summary,
    }

    # Save Evaluation Artifact
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("==================================================")
    print(" EVALUATION SUMMARY RESULTS ")
    print("==================================================")
    print(f" Total Test Cases Evaluated : {total_cases}")
    print(f" Intent Routing Accuracy    : {overall_intent_accuracy * 100:.1f}%")
    print(f" Retrieval Hit Rate (Hit@K) : {overall_hit_rate * 100:.1f}%")
    print(f" Lexical F1 Score   : {overall_mean_f1:.4f}")
    print(f" Semantic Similarity   : {overall_mean_semantic:.4f}")
    print(f"\nFull benchmark report exported to: {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()