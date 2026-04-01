import torch
import torch.nn as nn
from collections import defaultdict

def ctc_beam_search(log_probs, out_lengths, beam_size=5, blank_id=0):
    """
    Standard CTC Beam Search decoding implementation.
    log_probs: Tensor of shape (batch, time, vocab)
    out_lengths: Tensor of shape (batch,)
    
    Returns: list of decoded ID lists
    """
    batch_size = log_probs.size(0)
    decoded = []
    
    for b in range(batch_size):
        length = out_lengths[b].item()
        probs = torch.exp(log_probs[b, :length]) # (time, vocab)
        
        # Beam search initialization
        # beam is a dict of {tuple(sequence): (prob_blank, prob_non_blank)}
        beam = {tuple(): (1.0, 0.0)}
        
        for t in range(length):
            next_beam = defaultdict(lambda: (0.0, 0.0))
            for seq, (p_b, p_nb) in beam.items():
                p_total = p_b + p_nb
                
                # Case 1: blank extension
                next_beam[seq] = (
                    next_beam[seq][0] + p_total * probs[t, blank_id].item(),
                    next_beam[seq][1]
                )
                
                # Case 2: non-blank extension
                for c in range(1, probs.size(1)):
                    p_c = probs[t, c].item()
                    
                    if len(seq) > 0 and c == seq[-1]:
                        # Repeated token: needs blank in between to be considered new
                        next_beam[seq][1] += p_nb * p_c
                        new_seq = seq + (c,)
                        next_beam[new_seq] = (
                            next_beam[new_seq][0],
                            next_beam[new_seq][1] + p_b * p_c
                        )
                    else:
                        new_seq = seq + (c,)
                        next_beam[new_seq] = (
                            next_beam[new_seq][0],
                            next_beam[new_seq][1] + p_total * p_c
                        )
            
            # Keep top-k beams
            beam = dict(sorted(next_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_size])
            
        best_seq = max(beam.items(), key=lambda x: x[1][0] + x[1][1])[0]
        decoded.append(list(best_seq))
        
    return decoded
