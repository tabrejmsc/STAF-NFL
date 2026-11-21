"""The three spike-domain benchmarks of Table I, built deterministically from a fixed seed.

ND-Digits   UCI optical handwritten digits (sklearn), Poisson rate coding, T=20.
DVS-Gest-S  simulated DVS gestures on a 16x16 sensor (ON/OFF -> 512 channels), T=25.
            Gesture trajectories are rendered as intensity frames (hand blob + forearm
            over a static textured background) and converted to events with the
            standard log-intensity threshold-crossing model, including per-pixel
            threshold mismatch, background activity and hot pixels.
IoT-Event   synthetic multivariate machine-health signals (16 sensors, 5 condition
            classes) encoded by a level-crossing (delta-modulation) ADC into
            UP/DOWN channels (32 channels), T=30.

The paper does not publish its generators for the two simulated corpora, so these
are re-implementations that match the published sizes (N, D, T, C, d) and spike
densities of Table I; their absolute accuracies are therefore not expected to equal
the paper's to the decimal.
"""
from __future__ import annotations

import os

import numpy as np

DATA_SEED = 20250  # fixed: all benchmarks are a deterministic function of this seed

BENCHMARKS = {
    "ND-Digits": dict(T=20, hidden=(256,), rounds=60),
    "DVS-Gest-S": dict(T=25, hidden=(300, 150), rounds=100),
    "IoT-Event": dict(T=30, hidden=(256,), rounds=90),
}


# ---------------------------------------------------------------- ND-Digits
def make_nd_digits(T=20, max_rate=0.6, seed=DATA_SEED):
    from sklearn.datasets import load_digits
    d = load_digits()
    x = d.data.astype(np.float64) / 16.0                     # (1797, 64) in [0, 1]
    rng = np.random.default_rng(seed)
    p = max_rate * x                                         # Poisson (Bernoulli) rate per step
    X = rng.random((x.shape[0], T, x.shape[1])) < p[:, None, :]
    return X.astype(np.uint8), d.target.astype(np.int64)


# --------------------------------------------------------------- DVS-Gest-S
DVS_CLASSES = ["swipe-right", "swipe-left", "swipe-down", "swipe-up", "circle-cw",
               "circle-ccw", "diag-down", "diag-up", "figure-eight", "wave"]


def _trajectory(c, u, phase):
    """Unit-scale gesture path (x, y), y pointing down, as a function of u in [0,1]."""
    two_pi = 2 * np.pi
    if c == 0:   return -1 + 2 * u, 0 * u
    if c == 1:   return 1 - 2 * u, 0 * u
    if c == 2:   return 0 * u, -1 + 2 * u
    if c == 3:   return 0 * u, 1 - 2 * u
    if c == 4:   return np.cos(two_pi * u + phase), np.sin(two_pi * u + phase)
    if c == 5:   return np.cos(-two_pi * u + phase), np.sin(-two_pi * u + phase)
    if c == 6:   return 0.8 * (-1 + 2 * u), 0.8 * (-1 + 2 * u)
    if c == 7:   return 0.8 * (1 - 2 * u), 0.8 * (1 - 2 * u)
    if c == 8:   return np.sin(two_pi * u), 0.6 * np.sin(2 * two_pi * u)
    if c == 9:   return 0.8 * np.sin(2 * two_pi * u), 0.15 * np.cos(2 * two_pi * u)
    raise ValueError(c)


def make_dvs_gest(n=3000, T=25, res=16, sub=4, C_thr=0.085, mismatch=0.06,
                  bg_rate=0.02, hot_frac=0.01, hot_rate=0.3, rot_deg=40.0, distractor_p=0.65,
                  seed=DATA_SEED + 1):
    rng = np.random.default_rng(seed)
    F = T * sub + 1                                          # internal frames
    g = np.arange(res) - (res - 1) / 2.0
    xx, yy = np.meshgrid(g, g)                               # (res, res)
    # fixed sensor properties (one physical sensor for the whole corpus)
    C_on = np.clip(C_thr * (1 + mismatch * rng.standard_normal((res, res))), 0.03, None)
    C_off = np.clip(C_thr * (1 + mismatch * rng.standard_normal((res, res))), 0.03, None)
    hot = rng.random((2, res, res)) < hot_frac
    y = np.repeat(np.arange(10), int(np.ceil(n / 10)))[:n]
    rng.shuffle(y)
    X = np.zeros((n, T, 2, res, res), dtype=np.uint8)
    tau = np.linspace(0, 1, F)
    for i in range(n):
        c = y[i]
        t0 = rng.uniform(0.0, 0.25)
        dur = min(rng.uniform(0.55, 0.95), 1.0 - t0)
        u = np.clip((tau - t0) / dur, 0, 1)
        px, py = _trajectory(c, u, -np.pi / 2 + rng.uniform(-0.6, 0.6))
        A = rng.uniform(3.5, 5.5)
        th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
        cx, cy = rng.uniform(-1.5, 1.5, size=2)
        hx = cx + A * (np.cos(th) * px - np.sin(th) * py)    # (F,)
        hy = cy + A * (np.sin(th) * px + np.cos(th) * py)
        sig = rng.uniform(1.3, 2.0)
        k = rng.uniform(1.0, 2.0)
        # static, low-contrast textured background (smooth random field)
        bgf = rng.standard_normal((res, res))
        bgf = (bgf + np.roll(bgf, 1, 0) + np.roll(bgf, 1, 1) + np.roll(bgf, -1, 0) + np.roll(bgf, -1, 1)) / 5
        B = 1.0 + 0.15 * bgf
        # hand blob
        d2 = (xx[None] - hx[:, None, None]) ** 2 + (yy[None] - hy[:, None, None]) ** 2
        Ihand = k * np.exp(-d2 / (2 * sig ** 2))
        # forearm: segment from an elbow below the frame to the hand
        ex, ey = 0.5 * hx, np.full(F, res / 2 + 2.0)
        vx, vy = hx - ex, hy - ey
        L2 = vx ** 2 + vy ** 2 + 1e-9
        tt = np.clip(((xx[None] - ex[:, None, None]) * vx[:, None, None] +
                      (yy[None] - ey[:, None, None]) * vy[:, None, None]) / L2[:, None, None], 0, 1)
        dx = xx[None] - (ex[:, None, None] + tt * vx[:, None, None])
        dy = yy[None] - (ey[:, None, None] + tt * vy[:, None, None])
        Iarm = 0.6 * k * np.exp(-(dx ** 2 + dy ** 2) / 2.0)
        I = B[None] + np.maximum(Ihand, Iarm)
        if rng.random() < distractor_p:                      # unrelated moving object
            p0 = rng.uniform(-res / 2, res / 2, size=2)
            v = rng.normal(0, 4.0, size=2)
            ox, oy = p0[0] + v[0] * tau, p0[1] + v[1] * tau
            od2 = (xx[None] - ox[:, None, None]) ** 2 + (yy[None] - oy[:, None, None]) ** 2
            I = I + rng.uniform(0.3, 0.8) * np.exp(-od2 / (2 * rng.uniform(1.0, 2.0) ** 2))
        I = I * np.exp(0.01 * rng.standard_normal(I.shape))  # photoreceptor noise
        Lg = np.log(I)
        ref = Lg[0].copy()
        on = np.zeros((F - 1, res, res), dtype=bool)
        off = np.zeros((F - 1, res, res), dtype=bool)
        for f in range(1, F):
            d = Lg[f] - ref
            n_on = np.floor(np.maximum(d, 0) / C_on)
            n_off = np.floor(np.maximum(-d, 0) / C_off)
            on[f - 1] = n_on > 0
            off[f - 1] = n_off > 0
            ref = ref + n_on * C_on - n_off * C_off
        on = on.reshape(T, sub, res, res).any(axis=1)
        off = off.reshape(T, sub, res, res).any(axis=1)
        ev = np.stack([on, off], axis=1)                     # (T, 2, res, res)
        ev |= rng.random(ev.shape) < bg_rate                 # background activity
        ev |= hot[None] & (rng.random(ev.shape) < hot_rate)  # hot pixels
        X[i] = ev
    return X.reshape(n, T, 2 * res * res), y.astype(np.int64)


# ---------------------------------------------------------------- IoT-Event
IOT_CLASSES = ["normal", "imbalance", "misalignment", "bearing-fault", "looseness"]


def make_iot_event(n=3000, T=30, n_sensors=16, sub=8, delta=0.095, noise=0.08, seed=DATA_SEED + 2):
    rng = np.random.default_rng(seed)
    t = np.arange(T * sub) / sub                             # time in encoder steps
    S = 6                                                    # latent sources
    mix = rng.normal(0, 1, size=(n_sensors, S)) / np.sqrt(S)  # fixed sensor placement
    y = np.repeat(np.arange(5), int(np.ceil(n / 5)))[:n]
    rng.shuffle(y)
    X = np.zeros((n, T, 2 * n_sensors), dtype=np.uint8)
    for i in range(n):
        c = y[i]
        f0 = rng.uniform(0.07, 0.14)                         # shaft rotation, cycles / step
        sev = rng.uniform(0.15, 1.0)                         # fault severity
        ph = rng.uniform(0, 2 * np.pi, size=6)
        src = np.zeros((S, t.size))
        src[0] = 0.6 * np.sin(2 * np.pi * f0 * t + ph[0])                 # 1x
        src[1] = 0.25 * np.sin(2 * np.pi * 2 * f0 * t + ph[1])            # 2x
        src[2] = 0.1 * np.sin(2 * np.pi * 3 * f0 * t + ph[2])             # 3x
        drift = rng.uniform(-0.5, 0.5)
        src[5] = drift * (t / t[-1]) + 0.3 * np.sin(2 * np.pi * 0.01 * t + ph[5])  # thermal
        if c == 1:      # imbalance: strong 1x
            src[0] *= 1 + 2.0 * sev
        elif c == 2:    # misalignment: strong 2x, axial component in anti-phase
            src[1] *= 1 + 4.0 * sev
            src[3] = 0.8 * sev * np.sin(2 * np.pi * 2 * f0 * t + ph[1] + np.pi)
        elif c == 3:    # bearing fault: periodic impulses exciting a resonance
            bpf = f0 * rng.uniform(3.0, 3.6)                 # ball-pass frequency
            fr = rng.uniform(0.35, 0.45)                     # resonance, cycles / step
            t_imp = np.arange(rng.uniform(0, 1 / bpf), t[-1], 1 / bpf)
            for ti in t_imp:
                dtt = t - ti
                m = dtt >= 0
                src[4, m] += 1.6 * sev * np.exp(-dtt[m] / 0.8) * np.sin(2 * np.pi * fr * dtt[m])
        elif c == 4:    # mechanical looseness: sub-harmonic, many harmonics, rattling
            src[3] = 0.6 * sev * np.sin(np.pi * f0 * t + ph[3])
            for h in range(2, 6):
                src[3] += 0.25 * sev / h * np.sin(2 * np.pi * h * f0 * t + ph[4] * h)
            n_rat = rng.poisson(3 * sev) + 1
            for ti in rng.uniform(0, t[-1], size=n_rat):
                dtt = t - ti
                m = dtt >= 0
                src[4, m] += 1.0 * sev * np.exp(-dtt[m] / 0.5) * rng.choice([-1, 1])
        gains = mix * (1 + 0.1 * rng.standard_normal(mix.shape))
        # sensor noise: band-limited (AR(1), one-step correlation) plus a small white floor
        w = rng.standard_normal((n_sensors, t.size))
        ar = np.empty_like(w)
        ar[:, 0] = w[:, 0]
        a = np.exp(-1.0 / sub)
        for j in range(1, t.size):
            ar[:, j] = a * ar[:, j - 1] + np.sqrt(1 - a * a) * w[:, j]
        sig = gains @ src + noise * ar + 0.01 * rng.standard_normal((n_sensors, t.size))
        # level-crossing (delta-modulation) ADC with a tracking reference:
        # UP when the input rises delta above the reference, DOWN when it falls delta below
        ref = sig[:, 0].copy()
        up = np.zeros((n_sensors, t.size), dtype=bool)
        dn = np.zeros((n_sensors, t.size), dtype=bool)
        for j in range(1, t.size):
            d = sig[:, j] - ref
            nu = np.floor(np.maximum(d, 0) / delta)
            nd = np.floor(np.maximum(-d, 0) / delta)
            up[:, j] = nu > 0
            dn[:, j] = nd > 0
            ref += (nu - nd) * delta
        up = up.reshape(n_sensors, T, sub).any(axis=2)
        dn = dn.reshape(n_sensors, T, sub).any(axis=2)
        X[i] = np.concatenate([up, dn], axis=0).T
    return X, y.astype(np.int64)


# -------------------------------------------------------------- dispatch
_GEN = {"ND-Digits": make_nd_digits, "DVS-Gest-S": make_dvs_gest, "IoT-Event": make_iot_event}
_MEM = {}


def load_benchmark(name, cache_dir=None, test_size=0.25):
    """Return dict with X_train, y_train, X_test, y_test (uint8 spikes, int labels)."""
    if name in _MEM:
        return _MEM[name]
    cache_dir = cache_dir or os.environ.get("STAFNFL_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path)
        X, y = z["X"], z["y"]
    else:
        X, y = _GEN[name](T=BENCHMARKS[name]["T"])
        np.savez_compressed(path, X=X, y=y)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=DATA_SEED)
    ds = dict(name=name, X_train=X[tr], y_train=y[tr], X_test=X[te], y_test=y[te],
              T=X.shape[1], D=X.shape[2], C=int(y.max()) + 1, N=len(y),
              density=float(X.mean()), hidden=BENCHMARKS[name]["hidden"],
              rounds=BENCHMARKS[name]["rounds"])
    _MEM[name] = ds
    return ds
