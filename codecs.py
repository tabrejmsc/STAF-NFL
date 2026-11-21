"""Uplink codecs and their bit-level payload model (Section III-C, Table II).

All codecs act per tensor l (d_l coordinates). Bits are charged as follows:
  dense FP32      32 d_l
  8-bit (FedPAQ)  8 d_l + 32                     (symmetric, one FP32 scale per tensor)
  Top-k FP32      k_l (32 + ceil(log2 d_l))      (value + random-access index)
  SSC (ours)      k_l (1 + ceil(log2 d_l)) + 32  (sign bit + index, one FP32 scale)
with k_l = max(1, round(s d_l)). The optional Golomb (Rice) variant codes the gaps
between retained indices instead of charging ceil(log2 d_l) bits per index.
"""
from __future__ import annotations

import math

import numpy as np


def k_of(d, s):
    return max(1, int(np.rint(s * d)))


def idx_bits(d):
    return int(math.ceil(math.log2(d))) if d > 1 else 1


def _topk_mask(x, k):
    flat = np.abs(x).ravel()
    if k >= flat.size:
        return np.ones(x.shape, dtype=bool)
    thr_idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros(flat.size, dtype=bool)
    m[thr_idx] = True
    return m.reshape(x.shape)


# ----------------------------------------------------------------- encoders
def encode_dense(P):
    return [p.copy() for p in P], sum(32 * p.size for p in P)


def encode_int8(P):
    out, bits = [], 0
    for p in P:
        a = float(np.max(np.abs(p)))
        scale = a / 127.0 if a > 0 else 1.0
        q = np.clip(np.rint(p / scale), -127, 127)
        out.append((q * scale).astype(p.dtype))
        bits += 8 * p.size + 32
    return out, bits


def encode_topk(P, s):
    out, bits = [], 0
    for p in P:
        k = k_of(p.size, s)
        m = _topk_mask(p, k)
        out.append(np.where(m, p, 0).astype(p.dtype))
        bits += k * (32 + idx_bits(p.size))
    return out, bits


def encode_ssc(P, s):
    out, bits = [], 0
    for p in P:
        k = k_of(p.size, s)
        m = _topk_mask(p, k)
        alpha = float(np.abs(p[m]).mean()) if k > 0 else 0.0
        out.append((alpha * np.sign(p) * m).astype(p.dtype))
        bits += k * (1 + idx_bits(p.size)) + 32
    return out, bits


def encode(P, codec, s=0.05):
    if codec == "dense":
        return encode_dense(P)
    if codec == "int8":
        return encode_int8(P)
    if codec == "topk":
        return encode_topk(P, s)
    if codec == "ssc":
        return encode_ssc(P, s)
    raise ValueError(codec)


# ----------------------------------------------------- Golomb / Rice variant
def rice_bits_for_mask(mask):
    """Bits to code the retained indices of one tensor as Rice-coded gaps,
    with the Rice parameter chosen optimally (5 header bits to send it)."""
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return 0
    gaps = np.diff(np.concatenate([[-1], idx])) - 1
    best = None
    for b in range(0, 24):
        tot = int(np.sum((gaps >> b) + 1 + b))
        best = tot if best is None else min(best, tot)
    return best + 5


def golomb_bits(P, codec, s=0.05):
    bits = 0
    for p in P:
        if codec == "dense":
            bits += 32 * p.size
        elif codec == "int8":
            bits += 8 * p.size + 32
        else:
            k = k_of(p.size, s)
            m = _topk_mask(p, k)
            ib = rice_bits_for_mask(m)
            bits += (k * 32 + ib) if codec == "topk" else (k + ib + 32)
    return bits
