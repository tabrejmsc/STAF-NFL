"""Federated simulation: baselines and the STAF-NFL mechanisms (STA, EDT, SSC).

A single routine, :func:`run_federated`, implements every method in the paper as a
combination of switches, so that each component can be enabled in isolation (the
ablation of Table IV) and so that the FedAvg limit is reached by construction:

  sta=None,  edt=False, codec='dense'          -> FedAvg
  sta='timing', rho=0                          -> FedAvg (bitwise-identical)
  edt=True,  theta0 <= 1                       -> FedAvg (bitwise-identical)
"""
from __future__ import annotations

import time

import numpy as np

from . import codecs
from .data import load_benchmark
from .snn import SNN, timing_scores

# --------------------------------------------------------------------- config
DEFAULTS = dict(
    n_clients=20, alpha=0.5, rounds=None, local_epochs=1,
    lr=0.05, momentum=0.9, batch_size=32,
    # STAF-NFL defaults (Section IV)
    rho=0.7, kappa=3.0, s=0.05, theta0=2.5, lam=0.95, tau_max=6,
    # component switches
    sta=None,            # None | 'timing' | 'rate'
    edt=False,
    codec="dense",       # 'dense' | 'int8' | 'topk' | 'ssc'
    ef=False,            # error feedback
    prox_mu=0.0,
    scaffold=False, scaffold_momentum_fix=True,
    # EDT design alternatives discussed in Sections III-B and V
    edt_reset="soft",    # 'soft' | 'hard'
    edt_norm="current",  # 'current' (paper) | 'ema' (earlier formulation)
    edt_ema=0.9,
    # SNN hyper-parameters
    beta=0.9, beta_out=0.9, v_th=1.0, slope=5.0, init_gain=1.5,
    eval_every=1,
)

METHODS = {
    # main comparison (Table III)
    "FedAvg": dict(),
    "FedProx": dict(prox_mu=0.05),
    "SCAFFOLD": dict(scaffold=True),
    "FedPAQ": dict(codec="int8"),
    "Top-k+EF": dict(codec="topk", ef=True),
    "FedSNN-R": dict(sta="rate"),
    "STAF-NFL": dict(sta="timing", edt=True, codec="ssc", ef=True),
    # ablation (Table IV)
    "STA only": dict(sta="timing"),
    "EDT only": dict(edt=True),
    "SSC only": dict(codec="ssc", ef=True),
    "STA+SSC": dict(sta="timing", codec="ssc", ef=True),
    "STA+EDT": dict(sta="timing", edt=True),
    "EDT+SSC": dict(edt=True, codec="ssc", ef=True),
}

ENERGY = dict(e_sop=23.6e-12, e_mac=4.6e-12, e_tx=50e-9, link_bps=250e3)


# ------------------------------------------------------------------ partition
def dirichlet_partition(y, n_clients, alpha, rng, min_size=None, max_tries=1000):
    """Label-skew partition of Hsu et al.: for each class draw client proportions
    from Dir(alpha) and split that class's indices accordingly."""
    n, C = len(y), int(y.max()) + 1
    if min_size is None:
        min_size = max(2, min(10, n // (4 * n_clients)))
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in range(C):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    else:
        # severe skew / many clients: top up small clients from the largest ones
        for k in range(n_clients):
            while len(parts[k]) < min_size:
                big = int(np.argmax([len(p) for p in parts]))
                parts[k].append(parts[big].pop(int(rng.integers(len(parts[big])))))
    return [np.array(sorted(p), dtype=np.int64) for p in parts]


# ------------------------------------------------------------------- helpers
def _norm(ts):
    return float(np.sqrt(sum(float(np.dot(t.ravel().astype(np.float64), t.ravel().astype(np.float64))) for t in ts)))


def local_train(net, theta, X, y, rng, cfg, corr=None, stats=None):
    params = [p.copy() for p in theta]
    vel = [np.zeros_like(p) for p in params]
    lr = np.float32(cfg["lr"])
    mom = np.float32(cfg["momentum"])
    mu = np.float32(cfg["prox_mu"])
    bs = cfg["batch_size"]
    n = len(y)
    steps = 0
    for _ in range(cfg["local_epochs"]):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            _, g = net.loss_and_grad(params, X[b], y[b], stats=stats)
            for j in range(len(params)):
                gj = g[j]
                if mu:
                    gj = gj + mu * (params[j] - theta[j])
                if corr is not None:
                    gj = gj + corr[j]
                vel[j] *= mom
                vel[j] += gj
                params[j] -= lr * vel[j]
            steps += 1
    return params, steps


def sta_weights(net, phis, nks, rho, kappa):
    """Eq. (2). phis: per participating client, list over hidden layers of (H_l,)
    scores. Returns per client a list (one entry per parameter tensor) of weights
    w_{k,:} broadcastable against the tensor, plus the raw psi for diagnostics."""
    nks = np.asarray(nks, dtype=np.float64)
    nF = nks.sum()
    L = net.n_hidden
    psi = []
    for l in range(L):
        Phi = np.stack([p[l] for p in phis])                     # (|F|, H_l)
        phibar = (nks[:, None] * Phi).sum(axis=0) / nF
        ok = phibar > 0
        tilde = np.where(ok, Phi / np.where(ok, phibar, 1.0), 1.0)
        tilde = np.clip(tilde, 1.0 / kappa, kappa)
        psi.append((1.0 - rho) + rho * tilde)
    out = []
    for k in range(len(nks)):
        base = nks[k] / nF
        w = []
        for l in range(L):
            w.append(base * psi[l][k][None, :])                  # W_l: output dimension
            w.append(base * psi[l][k])                           # b_l
        w.append(base * psi[L - 1][k][:, None])                  # read-out W: input dimension
        w.append(np.asarray(base * psi[L - 1][k].mean()))        # read-out bias: scalar companion
        out.append(w)
    return out, psi


def aggregate(theta, payloads, weights):
    """theta + sum_k w_k (.) P_k, accumulated in float64 in client order."""
    agg = [np.zeros(p.shape, dtype=np.float64) for p in theta]
    for P, W in zip(payloads, weights):
        for i in range(len(theta)):
            agg[i] += np.asarray(W[i], dtype=np.float64) * P[i]
    return [(theta[i].astype(np.float64) + agg[i]).astype(theta[i].dtype) for i in range(len(theta))]


# ------------------------------------------------------------------ main loop
def run_federated(bench, method="FedAvg", seed=0, verbose=False, return_params=False,
                  data=None, **overrides):
    cfg = dict(DEFAULTS)
    cfg.update(METHODS[method] if method in METHODS else {})
    cfg.update(overrides)
    ds = data or load_benchmark(bench)
    R = cfg["rounds"] or ds["rounds"]
    N = cfg["n_clients"]
    sizes = (ds["D"],) + tuple(ds["hidden"]) + (ds["C"],)
    net = SNN(sizes, T=ds["T"], beta=cfg["beta"], beta_out=cfg["beta_out"], v_th=cfg["v_th"],
              slope=cfg["slope"], init_gain=cfg["init_gain"])
    d = net.n_params
    theta = net.init_params(np.random.default_rng([seed, 0]))
    Xtr, ytr, Xte, yte = ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"]
    parts = dirichlet_partition(ytr, N, cfg["alpha"], np.random.default_rng([seed, 1]))
    nk = np.array([len(p) for p in parts], dtype=np.float64)
    crng = [np.random.default_rng([seed, 1000 + k]) for k in range(N)]

    zeros = lambda: [np.zeros_like(p) for p in theta]
    st_A = [zeros() for _ in range(N)] if cfg["edt"] else None
    st_m = np.zeros(N)
    st_V = np.zeros(N)
    st_sil = np.zeros(N, dtype=int)
    st_ema = np.full(N, np.nan)
    st_R = [zeros() for _ in range(N)] if cfg["ef"] else None
    if cfg["scaffold"]:
        c_glob = zeros()
        c_loc = [zeros() for _ in range(N)]

    dense_bits = 32 * d
    hist = dict(round=[], acc=[], up_bytes=[], tx=[])
    up_bits_tot = 0.0
    down_bits_tot = 0.0
    latency = 0.0
    sops = 0.0
    n_samples_trained = 0
    tx_total = 0
    t_start = time.time()
    for r in range(R):
        payloads, weights_n, phis, round_bits = [], [], [], []
        dcs = []
        for k in range(N):
            stats = {} if cfg["sta"] else None
            idx = parts[k]
            corr = None
            if cfg["scaffold"]:
                corr = [cg - cl for cg, cl in zip(c_glob, c_loc[k])]
            local_stats = stats if stats is not None else {}
            new, steps = local_train(net, theta, Xtr[idx], ytr[idx], crng[k], cfg, corr=corr,
                                     stats=local_stats)
            sops += local_stats.get("sops", 0.0)
            n_samples_trained += local_stats.get("n", 0)
            delta = [a - b for a, b in zip(new, theta)]
            bits = 0
            if cfg["scaffold"]:
                coef = (1.0 - cfg["momentum"]) if cfg["scaffold_momentum_fix"] else 1.0
                coef = np.float32(coef / (steps * cfg["lr"]))
                c_new = [cl - cg - coef * dl for cl, cg, dl in zip(c_loc[k], c_glob, delta)]
                dcs.append([a - b for a, b in zip(c_new, c_loc[k])])
                c_loc[k] = c_new
                bits += dense_bits                                   # control-variate upload
            # ---------------- event-driven transmission (leaky integrate-and-fire)
            if cfg["edt"]:
                lam = np.float32(cfg["lam"])
                A = st_A[k]
                for i in range(len(A)):
                    A[i] = lam * A[i] + delta[i]
                st_m[k] = cfg["lam"] * st_m[k] + 1.0
                nd = _norm(delta)
                na = _norm(A)
                if cfg["edt_norm"] == "current":
                    inc = na / nd if nd > 0 else 0.0
                else:
                    st_ema[k] = nd if np.isnan(st_ema[k]) else cfg["edt_ema"] * st_ema[k] + (1 - cfg["edt_ema"]) * nd
                    inc = na / st_ema[k] if st_ema[k] > 0 else 0.0
                st_V[k] = cfg["lam"] * st_V[k] + inc
                fire = (st_V[k] >= cfg["theta0"]) or (st_sil[k] >= cfg["tau_max"])
                if not fire:
                    st_sil[k] += 1
                    continue
                mk = np.float32(st_m[k])
                P = [a / mk for a in A]
                st_A[k] = zeros()
                st_m[k] = 0.0
                st_V[k] = max(0.0, st_V[k] - cfg["theta0"]) if cfg["edt_reset"] == "soft" else 0.0
                st_sil[k] = 0
            else:
                P = delta
            # ---------------- codec with error feedback
            if cfg["ef"]:
                P = [p + rr for p, rr in zip(P, st_R[k])]
            Phat, b = codecs.encode(P, cfg["codec"], cfg["s"])
            if cfg["ef"]:
                st_R[k] = [p - ph for p, ph in zip(P, Phat)]
            bits += b
            payloads.append(Phat)
            weights_n.append(nk[k])
            round_bits.append(bits)
            if cfg["sta"]:
                phis.append([timing_scores(s_, local_stats["n"], cfg["sta"]) for s_ in local_stats["spk"]])
        # ---------------- server aggregation
        if payloads:
            nF = float(np.sum(weights_n))
            if cfg["sta"]:
                W, _ = sta_weights(net, phis, weights_n, cfg["rho"], cfg["kappa"])
            else:
                W = [[np.asarray(n_ / nF)] * len(theta) for n_ in weights_n]
            theta = aggregate(theta, payloads, W)
        if cfg["scaffold"] and dcs:
            for i in range(len(c_glob)):
                c_glob[i] = c_glob[i] + np.mean([dc[i] for dc in dcs], axis=0).astype(c_glob[i].dtype)
        up_bits_tot += float(np.sum(round_bits))
        down_bits_tot += N * dense_bits                              # dense broadcast to all clients
        latency += ((max(round_bits) if round_bits else 0.0) + dense_bits) / ENERGY["link_bps"]
        tx_total += len(payloads)
        if (r + 1) % cfg["eval_every"] == 0 or r == R - 1:
            acc = net.accuracy(theta, Xte, yte)
            hist["round"].append(r + 1)
            hist["acc"].append(acc)
            hist["up_bytes"].append(up_bits_tot / 8.0)
            hist["tx"].append(tx_total)
            if verbose:
                print(f"[{bench} {method} s{seed}] r{r + 1:3d} acc={acc * 100:6.2f} "
                      f"up={up_bits_tot / 8e6:9.3f}MB tx={len(payloads):2d}/{N}", flush=True)
    macs = float(n_samples_trained) * net.backward_macs_per_sample()
    e = energy_breakdown(up_bits_tot, down_bits_tot, sops, macs)
    rec = dict(
        bench=bench, method=method, seed=seed, cfg={k: v for k, v in cfg.items()},
        rounds=R, n_params=d, n_clients=N, client_sizes=nk.astype(int).tolist(),
        final_acc=hist["acc"][-1], hist=hist,
        uplink_bytes=up_bits_tot / 8.0, downlink_bytes=down_bits_tot / 8.0,
        tx_opportunities=R * N, transmissions=tx_total, tx_rate=tx_total / (R * N),
        sops=sops, macs=macs, energy=e, latency_s=latency,
        wall_s=time.time() - t_start,
    )
    if return_params:
        rec["_theta"] = theta
    return rec


def energy_breakdown(up_bits, down_bits, sops, macs, e_tx=None):
    e_tx = ENERGY["e_tx"] if e_tx is None else e_tx
    comp = sops * ENERGY["e_sop"] + macs * ENERGY["e_mac"]
    up = up_bits * e_tx
    down = down_bits * e_tx / 2.0
    return dict(compute=comp, uplink=up, downlink=down, comm=up + down, total=comp + up + down)
