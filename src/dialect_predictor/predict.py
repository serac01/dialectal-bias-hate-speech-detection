"""
predict.py

Python 3 port of the AAE/dialect prediction model from:
  Blodgett, S. L., Green, L., & O'Connor, B. (2016).
  Demographic Dialectal Variation in Social Media: A Case Study of
  African-American English. EMNLP.

Original (Python 2) source: https://github.com/slanglab/twitteraae
Ported for this project — logic is unchanged, only Python 3 syntax fixes
(xrange -> range, unicode/decode removed, print_function).

Output: a 4-way posterior distribution over
  [African-American, Hispanic, Asian, White]
associated language usage, for a given tokenized text. This is NOT a claim
about the author's actual identity or ethnicity -- it is an estimate of
how much the text's word usage resembles each group's word usage, per the
paper's own caveats.
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_FILE = os.path.join(_HERE, "model_vocab.txt")
MODEL_FILE = os.path.join(_HERE, "model_count_table.txt")

_vocab = None
_w2num = None
_wordprobs = None
_K = 0


def load_model():
    """Idempotent. Loads vocab + count table into memory."""
    global _vocab, _w2num, _wordprobs, _K
    if _wordprobs is not None:
        return

    N_wk = np.loadtxt(MODEL_FILE)
    N_k = N_wk.sum(0)
    _K = len(N_k)
    _wordprobs = (N_wk + 1) / N_k

    with open(VOCAB_FILE, encoding="utf-8") as f:
        _vocab = [line.split("\t")[-1].strip() for line in f]
    _w2num = {w: i for i, w in enumerate(_vocab)}
    assert len(_vocab) == N_wk.shape[0]


def _infer_cvb0(invocab_tokens, alpha, numpasses):
    doclen = len(invocab_tokens)

    Qs = np.zeros((doclen, _K))
    for i in range(doclen):
        w = invocab_tokens[i]
        Qs[i, :] = _wordprobs[_w2num[w], :]
        Qs[i, :] /= Qs[i, :].sum()
    lik = Qs.copy()

    Q_k = Qs.sum(0)
    for _ in range(1, numpasses):
        for i in range(doclen):
            Q_k -= Qs[i, :]
            Qs[i, :] = lik[i, :] * (Q_k + alpha)
            Qs[i, :] /= Qs[i, :].sum()
            Q_k += Qs[i, :]

    Q_k /= Q_k.sum()
    return Q_k


def predict(tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
    """
    tokens: list of str (already tokenized).
    Returns a length-4 numpy array [African-American, Hispanic, Asian, White]
    proportions, or None if there isn't enough in-vocabulary signal.
    """
    if _wordprobs is None:
        load_model()

    if len(tokens) == 0:
        return None

    invocab_tokens = [w.lower() for w in tokens if w.lower() in _w2num]

    if len(invocab_tokens) < thresh1:
        return None
    if len(invocab_tokens) / len(tokens) < thresh2:
        return None

    return _infer_cvb0(invocab_tokens, alpha=alpha, numpasses=numpasses)
