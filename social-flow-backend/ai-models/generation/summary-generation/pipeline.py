"""
High-level pipeline to:
- prepare data
- build tokenizer/vocab
- train models (abstractive/extractive)
- evaluate and serve

This pipeline wires together earlier components into an end-to-end flow.
"""

from typing import Optional, List, Dict
import os
from .utils import set_seed, save_pickle, load_pickle, logger
from .dataset import AbstractiveSummaryDataset, ExtractiveSummaryDataset
from .trainer import AbstractiveTrainer, ExtractiveTrainer
from .models import TransformerSeq2Seq, SentenceScorer
from .summarizer import Summarizer
from .evaluation import rouge_scores, bleu_scores, length_stats
from .config import MODEL_DIR, DEVICE

class SummaryPipeline:
    def __init__(self, tokenizer=None, sentence_splitter=None, sentence_encoder=None):
        set_seed()
        self.tokenizer = tokenizer
        self.sentence_splitter = sentence_splitter
        self.sentence_encoder = sentence_encoder
        self.abstractive_model = None
        self.extractive_model = None

    def build_tokenizer_from_corpus(self, texts: List[str], min_freq: int = 2, max_size: int = 30000):
        """
        Builds a simple tokenizer from raw texts; for production use HuggingFace tokenizers.
        """
        # naive word count
        from collections import Counter
        freqs = Counter()
        for doc in texts:
            for w in doc.strip().split():
                freqs[w] += 1
        stoi = {}
        itos = {}
        # special tokens will be filled by SimpleTokenizer
        idx = 0
        for w, c in freqs.most_common(max_size):
            if c >= min_freq:
                stoi[w] = idx
                itos[idx] = w
                idx += 1
        from .utils import SimpleTokenizer
        self.tokenizer = SimpleTokenizer(stoi=stoi, itos=itos)
        save_pickle(self.tokenizer, "tokenizer.pkl")
        return self.tokenizer

    def train_abstractive(self, train_df, val_df=None, vocab_size=30000):
        """
        Train abstractive TransformerSeq2Seq on train_df (columns: document, summary).
        """
        if self.tokenizer is None:
            # build tokenizer from training data
            self.build_tokenizer_from_corpus(train_df['document'].tolist() + train_df['summary'].tolist())
        # create datasets
        train_ds = AbstractiveSummaryDataset(train_df, tokenizer=self.tokenizer, max_input_len=1024, max_target_len=200)
        val_ds = AbstractiveSummaryDataset(val_df, tokenizer=self.tokenizer, max_input_len=1024, max_target_len=200) if val_df is not None else None

        # instantiate model
        self.abstractive_model = TransformerSeq2Seq(vocab_size=len(self.tokenizer.stoi), embed_dim=HIDDEN_SIZE)
        trainer = AbstractiveTrainer(self.abstractive_model, tokenizer=self.tokenizer)
        trainer.train(train_ds, val_dataset=val_ds)
        # save model state
        save_path = os.path.join(MODEL_DIR, "abstractive_model.pt")
        torch.save(self.abstractive_model.state_dict(), save_path)
        logger.info(f"Saved abstractive model to {save_path}")
        return self.abstractive_model

    def train_extractive(self, train_df, val_df=None, sent_feat_dim=768):
        """
        Train extractive SentenceScorer.
        train_df expected to have 'document_sentences' (list of strings) and 'labels' (list of indices)
        sentence_encoder must be set to obtain sentence vectors
        """
        if self.sentence_encoder is None:
            raise RuntimeError("sentence_encoder must be provided for extractive training")

        train_ds = ExtractiveSummaryDataset(train_df, sentence_encoder=self.sentence_encoder)
        val_ds = ExtractiveSummaryDataset(val_df, sentence_encoder=self.sentence_encoder) if val_df is not None else None

        self.extractive_model = SentenceScorer(sent_feat_dim=sent_feat_dim)
        trainer = ExtractiveTrainer(self.extractive_model)
        trainer.train(train_ds, epochs=EPOCHS)
        save_path = os.path.join(MODEL_DIR, "extractive_model.pt")
        torch.save(self.extractive_model.state_dict(), save_path)
        logger.info(f"Saved extractive model to {save_path}")
        return self.extractive_model

    def build_summarizer(self):
        """
        Compose a Summarizer object for inference (requires models/tokenizer loaded).
        """
        summarizer = Summarizer(abstractive_model=self.abstractive_model, abstractive_tokenizer=self.tokenizer,
                                extractive_model=self.extractive_model, sentence_splitter=self.sentence_splitter,
                                sentence_encoder=self.sentence_encoder)
        return summarizer

    def evaluate_abstractive(self, references: List[str], hypotheses: List[str]) -> Dict:
        rouge = rouge_scores(references, hypotheses)
        bleu = bleu_scores(references, hypotheses)
        lengths = length_stats(hypotheses)
        res = {"rouge": rouge, "bleu": bleu, "lengths": lengths}
        logger.info(f"Evaluation results: {res}")
        return res
