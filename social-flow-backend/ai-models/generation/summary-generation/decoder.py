"""
Decoding strategies for abstractive summarization:
- greedy decoding
- beam search
- nucleus (top-p) sampling

Each decoder returns token id sequences which can be decoded by the tokenizer.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from .config import BEAM_SIZE, LENGTH_PENALTY, NO_REPEAT_NGRAM_SIZE, TOP_K, TOP_P, MAX_SUMMARY_LENGTH
from .utils import logger
import math

def greedy_decode(model, src_ids: torch.LongTensor, tokenizer, max_len: int = MAX_SUMMARY_LENGTH, device=None) -> List[int]:
    """
    Greedy decode: at each step pick highest probability token.
    src_ids: (1, src_len) or (batch, src_len) — we assume batch size 1 here for simplicity
    """
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.to(device)
        generated = [tokenizer.stoi[tokenizer.sos_token]]
        for step in range(max_len):
            decoder_input = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src_ids, decoder_input)  # (1, tgt_len, vocab)
            next_token_logits = logits[0, -1, :]
            next_id = torch.argmax(next_token_logits).item()

            # prevent repeating n-grams if required (simple heuristic isn't implemented here)
            generated.append(next_id)
            if next_id == tokenizer.stoi.get(tokenizer.eos_token):
                break
    return generated

def beam_search_decode(model, src_ids: torch.LongTensor, tokenizer, beam_size: int = BEAM_SIZE,
                       max_len: int = MAX_SUMMARY_LENGTH, length_penalty: float = LENGTH_PENALTY) -> List[int]:
    """
    Standard beam search implementation.
    Returns best hypothesis token id list.
    Note: this is a straightforward implementation; there are many production optimizations possible.
    """
    device = next(model.parameters()).device
    model.eval()
    sos = tokenizer.stoi[tokenizer.sos_token]
    eos = tokenizer.stoi[tokenizer.eos_token]
    with torch.no_grad():
        src_ids = src_ids.to(device)
        # each beam: (tokens_list, logprob)
        beams = [([sos], 0.0)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for tokens, score in beams:
                if tokens[-1] == eos:
                    completed.append((tokens, score))
                    continue
                decoder_input = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model(src_ids, decoder_input)  # (1, tgt_len, vocab)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # (vocab,)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for k in range(beam_size):
                    candidate_tokens = tokens + [int(topk_ids[k].item())]
                    candidate_score = score + float(topk_log_probs[k].item())
                    new_beams.append((candidate_tokens, candidate_score))
            # keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            # early stop if enough completed
            if len(completed) >= beam_size:
                break

        all_hyps = completed + beams
        # apply length penalty and pick best
        def lp(tokens, score):
            # length penalty from Google NMT
            length = len(tokens)
            return score / ( (5 + length) ** length_penalty / ((5 + 1) ** length_penalty) )

        best = max(all_hyps, key=lambda x: lp(x[0], x[1]))
        return best[0]

def top_k_top_p_sampling(model, src_ids: torch.LongTensor, tokenizer, max_len=MAX_SUMMARY_LENGTH, top_k: int = TOP_K, top_p: float = TOP_P, temperature: float = 1.0):
    """
    Stochastic decoding using top-k and nucleus (top-p) sampling.
    """
    device = next(model.parameters()).device
    model.eval()
    sos = tokenizer.stoi[tokenizer.sos_token]
    eos = tokenizer.stoi[tokenizer.eos_token]
    generated = [sos]
    with torch.no_grad():
        src_ids = src_ids.to(device)
        for _ in range(max_len):
            decoder_input = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src_ids, decoder_input)[0, -1, :] / max(temperature, 1e-6)
            # top-k
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)
            # nucleus top-p: filter tokens until cumulative prob >= top_p
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cutoff = (cumulative_probs >= top_p).nonzero(as_tuple=False)
            if cutoff.numel() > 0:
                cut_idx = cutoff[0].item()
                values = values[:cut_idx + 1]
                indices = indices[:cut_idx + 1]
                probs = probs[:cut_idx + 1]
                probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            next_token = int(indices[idx].item())
            generated.append(next_token)
            if next_token == eos:
                break
    return generated
