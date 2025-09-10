"""
Evaluation metrics for caption generation.
"""

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def evaluate_bleu(references, hypotheses):
    return corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses])


def evaluate_meteor(references, hypotheses):
    return sum(meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)) / len(hypotheses)


def evaluate_rouge(references, hypotheses):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return scores
