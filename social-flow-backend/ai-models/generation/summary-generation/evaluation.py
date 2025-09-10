"""
Evaluation metrics for summarization:
- ROUGE (1/2/L)
- BLEU (optional)
- Simple length/coverage metrics
"""

from typing import List, Dict
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .utils import logger

def rouge_scores(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Compute average ROUGE scores using google/rouge_score package.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        aggregator.add_scores(scores)
    result = aggregator.aggregate()
    # extract mid scores
    res = {}
    for k, v in result.items():
        res[k + "_precision"] = v.mid.precision
        res[k + "_recall"] = v.mid.recall
        res[k + "_fmeasure"] = v.mid.fmeasure
    return res

def bleu_scores(references: List[str], hypotheses: List[str]) -> float:
    """
    Return average BLEU (sentence-level with smoothing).
    """
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie))
    return sum(scores) / len(scores)

def length_stats(hypotheses: List[str]):
    lens = [len(h.split()) for h in hypotheses]
    return {"avg_len": sum(lens) / len(lens), "min_len": min(lens), "max_len": max(lens)}
