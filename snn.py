"""Discrete-time LIF spiking network trained by surrogate-gradient BPTT (pure NumPy).

Model (Section III of the paper)
--------------------------------
Hidden (spiking) layer l, hard reset, detached reset path:
    U_l[t] = beta * U_l[t-1] * (1 - S_l[t-1]) + W_l S_{l-1}[t] + b_l
    S_l[t] = Theta(U_l[t] - v_th)
    dS/dU  ~= (1 + slope * |U - v_th|)^-2            (fast-sigmoid surrogate)

Read-out: non-spiking leaky integrator whose potential is averaged over the window
    U_o[t] = beta_o * U_o[t-1] + W_o S_L[t] + b_o,     z = (1/T) sum_t U_o[t]
Expanding the recursion, a spike at time t enters z with weight
    c[t] = sum_{tau >= t} beta_o^(tau - t)            (decreasing in t)
so early spikes influence the logits more than late ones.

Because the network is feed-forward, every layer is simulated for all T steps
before the next one, so all matrix products are batched over (batch x time) and
only the element-wise membrane recursion loops over t.

``mode='relaxed'`` replaces the Heaviside output with the smooth function
f(x) = x / (1 + slope*|x|), whose exact derivative *is* the surrogate. It runs the
identical forward/backward code and is what the finite-difference check uses.
"""
from __future__ import annotations

import numpy as np


class SNN:
    def __init__(self, sizes, T, beta=0.9, beta_out=0.9, v_th=1.0, slope=5.0,
                 init_gain=1.5):
        self.sizes = tuple(int(s) for s in sizes)       # e.g. (64, 256, 10)
        self.T = int(T)
        self.beta = float(beta)
        self.beta_out = float(beta_out)
        self.v_th = float(v_th)
        self.slope = float(slope)
        self.init_gain = float(init_gain)
        self.n_layers = len(self.sizes) - 1              # weight layers
        self.n_hidden = self.n_layers - 1                # spiking layers
        t = np.arange(self.T)
        # c[t] = sum_{tau=t}^{T-1} beta_o^(tau-t)
        self.c = np.array([np.sum(self.beta_out ** np.arange(self.T - tt)) for tt in t])
        self.cw = self.c / self.T                        # weight of step t in z

    # ------------------------------------------------------------------ params
    @property
    def shapes(self):
        out = []
        for i in range(self.n_layers):
            out.append((self.sizes[i], self.sizes[i + 1]))
            out.append((self.sizes[i + 1],))
        return out

    @property
    def n_params(self):
        return int(sum(np.prod(s) for s in self.shapes))

    def init_params(self, rng, dtype=np.float32):
        params = []
        for i in range(self.n_layers):
            fan_in = self.sizes[i]
            a = self.init_gain * np.sqrt(3.0 / fan_in)
            params.append(rng.uniform(-a, a, size=(fan_in, self.sizes[i + 1])).astype(dtype))
            params.append(np.zeros(self.sizes[i + 1], dtype=dtype))
        return params

    # ----------------------------------------------------------------- forward
    def forward(self, params, X, mode="hard", stats=None):
        """X: (B, T, D) array (binary spikes). Returns z (B, C) and a cache.

        If ``stats`` is a dict it is updated in place with
          'spk'  : list of (T, H_l) spike-count sums per hidden layer (for STA)
          'n'    : number of samples seen
          'sops' : synaptic operations of the event-driven forward pass
        """
        dt = params[0].dtype
        X = np.asarray(X, dtype=dt)
        B, T, _ = X.shape
        beta, vth = dt.type(self.beta), dt.type(self.v_th)
        inp = X
        cache = []
        sops = float(X.sum()) * self.sizes[1]
        for l in range(self.n_hidden):
            W, b = params[2 * l], params[2 * l + 1]
            H = W.shape[1]
            I = (inp.reshape(B * T, -1) @ W).reshape(B, T, H) + b
            U = np.empty_like(I)
            S = np.empty_like(I)
            hard = np.empty(I.shape, dtype=bool)
            u = np.zeros((B, H), dtype=dt)
            keep = np.ones((B, H), dtype=dt) * beta      # beta * (1 - S[t-1]), S[-1] = 0
            for t in range(T):
                u = keep * u + I[:, t]
                U[:, t] = u
                h = u >= vth
                hard[:, t] = h
                keep = beta * (~h)
            if mode == "hard":
                S[...] = hard
            elif mode == "relaxed":
                x = U - vth
                S[...] = x / (1.0 + self.slope * np.abs(x))
            else:
                raise ValueError(mode)
            cache.append((inp, U, hard, S))
            inp = S
            if mode == "hard":       # each emitted spike drives fan-out synapses
                sops += float(S.sum()) * self.sizes[l + 2]
        Wo, bo = params[-2], params[-1]
        Sw = np.einsum("bth,t->bh", inp, self.cw.astype(dt))        # c-weighted spike sum
        z = Sw @ Wo + bo * dt.type(self.cw.sum())
        cache.append((inp, Sw))
        if stats is not None:
            if "spk" not in stats:
                stats["spk"] = [np.zeros((T, c[1].shape[2])) for c in cache[:-1]]
                stats["n"] = 0
                stats["sops"] = 0.0
            for l, (_, _, _, S) in enumerate(cache[:-1]):
                stats["spk"][l] += S.sum(axis=0)
            stats["n"] += B
            stats["sops"] += sops
        return z, cache

    # ---------------------------------------------------------------- backward
    def backward(self, params, cache, z, y):
        """Cross-entropy loss on z and gradients w.r.t. every parameter."""
        dt = params[0].dtype
        B = z.shape[0]
        zs = z - z.max(axis=1, keepdims=True)
        p = np.exp(zs)
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(p[np.arange(B), y] + 1e-12)))
        dz = p
        dz[np.arange(B), y] -= 1.0
        dz /= B
        grads = [None] * len(params)
        Wo = params[-2]
        S_last, Sw = cache[-1]
        grads[-2] = Sw.T @ dz
        grads[-1] = dz.sum(axis=0) * dt.type(self.cw.sum())
        dSw = dz @ Wo.T                                               # (B, H)
        dS = dSw[:, None, :] * self.cw.astype(dt)[None, :, None]      # (B, T, H)
        beta, vth = dt.type(self.beta), dt.type(self.v_th)
        for l in reversed(range(self.n_hidden)):
            inp, U, hard, _ = cache[l]
            Bn, T, H = U.shape
            sg = 1.0 / (1.0 + self.slope * np.abs(U - vth)) ** 2
            keep = beta * (~hard)
            dI = np.empty_like(U)
            carry = np.zeros((Bn, H), dtype=dt)
            for t in range(T - 1, -1, -1):
                du = dS[:, t] * sg[:, t] + carry           # dL/dU[t]
                dI[:, t] = du
                if t > 0:                                  # U[t] = beta(1-S[t-1]) U[t-1] + I[t]
                    carry = du * keep[:, t - 1]
            W = params[2 * l]
            dI2 = dI.reshape(Bn * T, H)
            grads[2 * l] = inp.reshape(Bn * T, -1).T @ dI2
            grads[2 * l + 1] = dI2.sum(axis=0)
            if l > 0:
                dS = (dI2 @ W.T).reshape(Bn, T, -1)
        return loss, grads

    def loss_and_grad(self, params, X, y, mode="hard", stats=None):
        z, cache = self.forward(params, X, mode=mode, stats=stats)
        return self.backward(params, cache, z, y)

    def loss(self, params, X, y, mode="hard"):
        z, _ = self.forward(params, X, mode=mode)
        zs = z - z.max(axis=1, keepdims=True)
        lp = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True))
        return float(-np.mean(lp[np.arange(len(y)), y]))

    def predict(self, params, X, batch=500):
        out = []
        for i in range(0, len(X), batch):
            z, _ = self.forward(params, X[i:i + batch])
            out.append(np.argmax(z, axis=1))
        return np.concatenate(out)

    def accuracy(self, params, X, y):
        return float(np.mean(self.predict(params, X) == y))

    # ------------------------------------------------------------ op counting
    def backward_macs_per_sample(self):
        """Dense surrogate-gradient backward pass: weight gradient + input
        gradient for every weight tensor at every step (conservative)."""
        return 2 * self.T * sum(self.sizes[i] * self.sizes[i + 1] for i in range(self.n_layers))


# ---------------------------------------------------------------------------
# spike-timing statistics (Eq. 1)
# ---------------------------------------------------------------------------
def timing_scores(spk_sum, n, kind="timing"):
    """spk_sum: (T, H) summed spike trains over n samples.

    kind='timing':  phi = r * exp(-mu/T) * (1 - H/log T)
    kind='rate'  :  phi = r                         (FedSNN-R baseline)
    """
    s = spk_sum / max(n, 1)                      # batch-averaged spike train s_j(t)
    T = s.shape[0]
    r = s.mean(axis=0)                           # firing rate
    if kind == "rate":
        return r
    tot = s.sum(axis=0)
    fired = tot > 0
    p = np.where(fired, s / np.where(fired, tot, 1.0), 0.0)
    t = np.arange(T)[:, None]
    mu = (t * p).sum(axis=0)                     # temporal centroid
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=0)
    phi = r * np.exp(-mu / T) * np.clip(1.0 - ent / np.log(T), 0.0, 1.0)
    return np.where(fired, phi, 0.0)
