import math
from collections import Counter
import torch
import numpy as np


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    (c, r) = stats[:2]
    log_bleu_prec = 0.
    for x, y in zip(stats[2::2], stats[3::2]):
        if x == 0:
            log_bleu_prec += math.log(1e-9)  # 避免 log(0)
        else:
            log_bleu_prec += math.log(float(x) / y)
    log_bleu_prec /= 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0.,0.,0.,0.,0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    exclude = {'<PAD>', '<EOS>', '<UNK>'}
    words = []
    for i in x:
        if isinstance(i, torch.Tensor):
            i = i.item()
        word = vocab.get(str(i), '<UNK>')
        if word == '<EOS>':
            break
        if word not in exclude:
            words.append(word)
    return ' '.join(words)
