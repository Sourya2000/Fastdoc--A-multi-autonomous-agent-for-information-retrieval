# metrics.py
import time
from typing import List, Tuple

def compute_mrr(retrieved_docs: List[Tuple], ground_truth: str) -> float:
    """Mean Reciprocal Rank"""
    for rank, (doc, _) in enumerate(retrieved_docs, start=1):
        if ground_truth.lower() in doc.page_content.lower():
            return 1 / rank
    return 0.0

def recall_at_k(retrieved_docs: List[Tuple], ground_truth: str, k: int = 4) -> float:
    """Recall@k"""
    return float(any(ground_truth.lower() in doc.page_content.lower() for doc, _ in retrieved_docs[:k]))

def llm_judge(llm, question: str, answer: str, contexts: List[str]) -> dict:
    """Optional LLM-as-judge"""
    prompt = f"""
You are an expert evaluator.
Question: {question}
Answer: {answer}
Context: {' '.join(contexts)}
Score the answer from 1 to 5 for correctness and grounding.
Respond as JSON: {{ "score": number, "reason": string }}
"""
    return llm.invoke(prompt).content

def measure_latency(func, *args, **kwargs):
    """Return (result, latency_in_seconds)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start
