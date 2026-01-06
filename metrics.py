# metrics.py
from typing import List, Dict, Tuple
import re
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# LLM as Judge - Improved
# -----------------------------
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1  # Add slight temperature for variability
)


def llm_as_judge(query: str, answer: str, context: List[str]) -> float:
    if not answer or answer.strip() == "":
        return 1.0  # Empty answer gets lowest score
    
    # Truncate context if too long
    context_str = "\n".join([f"[Context {i+1}]: {ctx[:200]}..." if len(ctx) > 200 else f"[Context {i+1}]: {ctx}" 
                            for i, ctx in enumerate(context[:3])])
    
    prompt = f"""You are an impartial evaluator of question-answering systems.

QUESTION: {query}

ANSWER PROVIDED: {answer}

CONTEXT PROVIDED (first 3 snippets):
{context_str if context_str else "No context provided"}

SCORING CRITERIA (1-5 scale):
1 - Completely wrong, irrelevant, or no answer
2 - Partially correct but missing key information or misleading
3 - Mostly correct but incomplete or contains minor errors
4 - Correct and complete with minor issues
5 - Perfect: Accurate, complete, well-supported by context

IMPORTANT: Be strict. Only give 5 for perfect answers. Consider:
- Factual accuracy based on context
- Completeness in addressing the question
- Relevance to the question asked
- Use of supporting evidence from context

Return ONLY a single number between 1 and 5. No explanation.
"""
    
    try:
        resp = judge_llm.invoke(prompt)
        content = resp.content.strip()
        
        # Extract number from response
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        if numbers:
            score = float(numbers[0])
            return max(1.0, min(5.0, score))  # Clamp between 1-5
        else:
            # Fallback: check for textual scores
            if "1" in content or "one" in content.lower():
                return 1.0
            elif "2" in content or "two" in content.lower():
                return 2.0
            elif "3" in content or "three" in content.lower():
                return 3.0
            elif "4" in content or "four" in content.lower():
                return 4.0
            else:
                return 3.0  # Default average
    except Exception as e:
        print(f"Judge error: {e}")
        return 3.0  # Default average on error


# -----------------------------
# Mean Reciprocal Rank (MRR) - Improved
# -----------------------------
def compute_mrr(retrieved_contexts: List[str], query: str) -> float:
    """
    Calculate MRR based on how well the QUERY (not answer) matches retrieved contexts.
    More realistic: check if the MOST RELEVANT context appears early.
    """
    if not retrieved_contexts:
        return 0.0
    
    # Clean query
    query_lower = query.lower().strip()
    if not query_lower:
        return 0.0
    
    # Remove common stopwords and get meaningful terms
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    query_terms = [term for term in re.findall(r'\b\w+\b', query_lower) 
                  if term not in stopwords and len(term) > 2]
    
    # If no meaningful terms, use all terms
    if not query_terms:
        query_terms = re.findall(r'\b\w+\b', query_lower)
    
    # Calculate term frequency in each context
    context_scores = []
    for context in retrieved_contexts:
        context_lower = str(context).lower()
        score = 0
        for term in query_terms:
            if term in context_lower:
                score += 1
        # Normalize by number of query terms
        normalized_score = score / len(query_terms) if query_terms else 0
        context_scores.append(normalized_score)
    
    # Find the position of the best-matching context
    if context_scores:
        best_score = max(context_scores)
        if best_score > 0:
            # Find first occurrence of best score
            for rank, score in enumerate(context_scores, 1):
                if score == best_score:
                    return 1.0 / rank
    
    return 0.0


# -----------------------------
# Alternative: Position-based MRR
# -----------------------------
def compute_mrr_position_based(retrieved_contexts: List[str], query: str, answer: str) -> float:
    """
    Alternative MRR: Check if answer-relevant terms appear in contexts.
    This assumes the answer should contain terms from the most relevant context.
    """
    if not retrieved_contexts or not answer:
        return 0.0
    
    # Extract key terms from answer
    answer_lower = answer.lower()
    answer_terms = [term for term in re.findall(r'\b\w+\b', answer_lower) 
                   if len(term) > 3][:5]  # Top 5 answer terms
    
    if not answer_terms:
        return 0.0
    
    # Find first context containing any answer term
    for rank, context in enumerate(retrieved_contexts, 1):
        context_lower = str(context).lower()
        for term in answer_terms:
            if term in context_lower:
                return 1.0 / rank
    
    return 0.0


# -----------------------------
# Simple RAGAS Evaluation
# -----------------------------
def compute_ragas(
    queries: List[str],
    answers: List[str],
    contexts: List[List[str]],
) -> Dict[str, float]:
    """
    Calculate simulated RAGAS metrics with more realistic scoring.
    """
    if not queries:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }
    
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    
    for query, answer, context_list in zip(queries, answers, contexts):
        # Get judge score for this query
        judge_score = llm_as_judge(query, answer, context_list)
        normalized = judge_score / 5.0
        
        # 1. Faithfulness: How well answer is grounded in context
        if context_list and answer:
            # Check if answer terms appear in context
            answer_terms = [t.lower() for t in re.findall(r'\b\w+\b', answer.lower()) if len(t) > 3]
            if answer_terms:
                matches = 0
                for ctx in context_list[:2]:  # Check top 2 contexts
                    ctx_lower = str(ctx).lower()
                    for term in answer_terms[:3]:  # Check top 3 terms
                        if term in ctx_lower:
                            matches += 1
                            break
                faithfulness = matches / min(2, len(context_list))
                faithfulness_scores.append(faithfulness * 0.7 + normalized * 0.3)
            else:
                faithfulness_scores.append(normalized * 0.8)
        else:
            faithfulness_scores.append(normalized * 0.5)
        
        # 2. Answer Relevancy: How relevant answer is to query
        if query and answer:
            # Simple term overlap between query and answer
            query_terms = set([t.lower() for t in re.findall(r'\b\w+\b', query.lower()) if len(t) > 2])
            answer_terms = set([t.lower() for t in re.findall(r'\b\w+\b', answer.lower()) if len(t) > 2])
            
            if query_terms and answer_terms:
                overlap = len(query_terms.intersection(answer_terms)) / len(query_terms)
                relevancy_scores.append(overlap * 0.6 + normalized * 0.4)
            else:
                relevancy_scores.append(normalized * 0.9)
        else:
            relevancy_scores.append(normalized * 0.7)
        
        # 3. Context Precision: How well contexts match query
        if context_list and query:
            query_terms = [t.lower() for t in re.findall(r'\b\w+\b', query.lower()) if len(t) > 2]
            if query_terms:
                precision_per_context = []
                for ctx in context_list[:3]:
                    ctx_lower = str(ctx).lower()
                    matches = sum(1 for term in query_terms[:5] if term in ctx_lower)
                    precision = matches / min(5, len(query_terms))
                    precision_per_context.append(precision)
                
                avg_precision = sum(precision_per_context) / len(precision_per_context) if precision_per_context else 0
                precision_scores.append(avg_precision * 0.8 + normalized * 0.2)
            else:
                precision_scores.append(normalized * 0.7)
        else:
            precision_scores.append(normalized * 0.5)
    
    # Calculate averages
    n = len(queries)
    return {
        "faithfulness": round(sum(faithfulness_scores) / n, 3) if faithfulness_scores else 0.0,
        "answer_relevancy": round(sum(relevancy_scores) / n, 3) if relevancy_scores else 0.0,
        "context_precision": round(sum(precision_scores) / n, 3) if precision_scores else 0.0,
    }


# -----------------------------
# Add this function to calculate MRR using the improved method
# -----------------------------
def compute_mrr_improved(retrieved_contexts: List[str], query: str, answer: str = "") -> float:
    """
    Combined MRR calculation - tries multiple methods and returns average.
    """
    mrr1 = compute_mrr(retrieved_contexts, query)
    
    if answer:
        mrr2 = compute_mrr_position_based(retrieved_contexts, query, answer)
        return (mrr1 + mrr2) / 2
    else:
        return mrr1