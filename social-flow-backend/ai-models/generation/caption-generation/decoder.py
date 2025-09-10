"""
Beam search and greedy decoding for generating captions.
"""

import torch
from .config import BEAM_SIZE, MAX_CAPTION_LEN


def greedy_decode(encoder, decoder, image, vocab):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        feature = encoder(image.unsqueeze(0))
        caption = [vocab.stoi["<SOS>"]]

        for _ in range(MAX_CAPTION_LEN):
            cap_tensor = torch.tensor([caption], device=image.device)
            output = decoder(feature, cap_tensor)
            next_word = output.argmax(-1)[:, -1].item()
            caption.append(next_word)
            if next_word == vocab.stoi["<EOS>"]:
                break

    return " ".join([vocab.itos[idx] for idx in caption[1:-1]])


def beam_search_decode(encoder, decoder, image, vocab, beam_size=BEAM_SIZE):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        feature = encoder(image.unsqueeze(0))
        sequences = [[list([vocab.stoi["<SOS>"]]), 0]]

        for _ in range(MAX_CAPTION_LEN):
            all_candidates = []
            for seq, score in sequences:
                cap_tensor = torch.tensor([seq], device=image.device)
                output = decoder(feature, cap_tensor)
                probs = torch.nn.functional.log_softmax(output[:, -1, :], dim=-1)
                topk = torch.topk(probs, beam_size)

                for idx, prob in zip(topk.indices[0], topk.values[0]):
                    candidate = [seq + [idx.item()], score + prob.item()]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_size]

        best_seq = sequences[0][0]
        return " ".join([vocab.itos[idx] for idx in best_seq[1:-1]])
