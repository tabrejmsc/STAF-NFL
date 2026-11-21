"""The experiment grid: which federated runs back which table / figure / number."""
from __future__ import annotations

BENCHES = ["ND-Digits", "DVS-Gest-S", "IoT-Event"]
MAIN_METHODS = ["FedAvg", "FedProx", "SCAFFOLD", "FedPAQ", "Top-k+EF", "FedSNN-R", "STAF-NFL"]
ABLATION = ["STA only", "EDT only", "SSC only", "STA+SSC", "STA+EDT", "EDT+SSC"]
HETERO_ALPHAS = [0.05, 0.1, 5.0]                 # alpha = 0.5 comes from the main suite
HETERO_METHODS = ["FedAvg", "Top-k+EF", "STA+SSC", "STAF-NFL"]
SENS = dict(rho=[0.0, 0.25, 0.5, 0.9, 1.0],      # default 0.7 from the main suite
            theta0=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0],  # default 2.5
            s=[0.01, 0.02, 0.1, 0.2])            # default 0.05
SCALE_N = [10, 50, 100]                          # N = 20 from the main suite
SUITES = ["main", "ablation", "hetero", "sens", "scale", "extras"]


def _tag(ov):
    return ",".join(f"{k}={v}" for k, v in sorted(ov.items())) or "default"


def jobs(suites, benches=None, seeds_main=5, seeds_sec=3, sens_benches=("ND-Digits",),
         scale_benches=("ND-Digits",), seeds_extras=3):
    benches = benches or BENCHES
    out = []

    def add(suite, b, m, seeds, ov=None):
        ov = ov or {}
        for s in range(seeds):
            out.append(dict(suite=suite, bench=b, method=m, seed=s, overrides=ov, tag=_tag(ov)))

    if "main" in suites:
        for b in benches:
            for m in MAIN_METHODS:
                add("main", b, m, seeds_main)
    if "ablation" in suites:
        for b in benches:
            for m in ABLATION:
                add("ablation", b, m, seeds_sec)
    if "hetero" in suites:
        for b in benches:
            for a in HETERO_ALPHAS:
                for m in HETERO_METHODS:
                    add("hetero", b, m, seeds_sec, dict(alpha=a))
            add("hetero", b, "STA+SSC", seeds_sec)           # alpha = 0.5 point for STA+SSC
    if "sens" in suites:
        for b in benches:
            if b not in sens_benches:
                continue
            for key, vals in SENS.items():
                for v in vals:
                    add("sens", b, "STAF-NFL", seeds_sec, {key: v})
    if "scale" in suites:
        for b in benches:
            if b not in scale_benches:
                continue
            for n in SCALE_N:
                for m in ["FedAvg", "STAF-NFL"]:
                    add("scale", b, m, seeds_sec, dict(n_clients=n))
    if "extras" in suites:
        # design alternatives the text argues against (Sections III-B, IV, V)
        for b in benches:
            if b == "ND-Digits":
                for th in [1.25, 1.5, 1.75, 1.95]:
                    add("extras", b, "STAF-NFL", seeds_extras, dict(edt_reset="hard", theta0=th))
                    add("extras", b, "STAF-NFL", seeds_extras, dict(edt_reset="soft", theta0=th))
                add("extras", b, "SCAFFOLD", seeds_extras, dict(scaffold_momentum_fix=False))
            add("extras", b, "STAF-NFL", 1, dict(edt_norm="ema"))
    # de-duplicate (the alpha=0.5 STA+SSC point duplicates the ablation suite)
    seen, uniq = set(), []
    for j in out:
        key = (j["bench"], j["method"], j["tag"], j["seed"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(j)
    return uniq


def job_filename(j):
    safe = lambda x: "".join(c if c.isalnum() or c in "-=.," else "_" for c in str(x))
    return f"{j['suite']}/{safe(j['bench'])}__{safe(j['method'])}__{safe(j['tag'])}__seed{j['seed']}.json"
