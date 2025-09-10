"""
Configuration constants for summary-generation package
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Training
DEVICE = "cuda"  # set to "cpu" if no GPU
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
EPOCHS = 6
MAX_INPUT_LENGTH = 1024    # for long inputs (document)
MAX_SUMMARY_LENGTH = 200
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 500

# Model
HIDDEN_SIZE = 768
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_HEADS = 12
FF_DIM = 2048
DROPOUT = 0.1
VOCAB_SIZE = 30000  # vocab fallback if building own tokenizer

# Decoding / generation
BEAM_SIZE = 4
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 3
TOP_K = 50
TOP_P = 0.95

# Evaluation
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]

# Misc
RANDOM_SEED = 42
