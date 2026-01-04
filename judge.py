# judge.py
from metrics import llm_judge

def evaluate_answer(llm, question: str, answer: str, contexts: list):
    """Return a dict with score and reason"""
    return llm_judge(llm, question, answer, contexts)
