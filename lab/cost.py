"""Budget ledger. Every GPU function records its own wall-clock cost on the shared Volume."""
import contextlib, glob, json, os, time

RATES = {  # USD per second (Modal list prices, checked Nov 2026)
    "T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097,
}
OVERHEAD = 1.06          # CPU + memory surcharge, rough
LEDGER = "/vol/ledger"


def spent(ledger=LEDGER):
    tot = 0.0
    for p in glob.glob(f"{ledger}/*.json"):
        try:
            tot += json.load(open(p))["usd"]
        except Exception:
            pass
    return tot


@contextlib.contextmanager
def track(vol, fn, gpu, cap, tag=""):
    os.makedirs(LEDGER, exist_ok=True)
    vol.reload()
    s = spent()
    if s >= cap:
        raise RuntimeError(f"Budget guard: ledger says ${s:.2f} >= cap ${cap:.2f}. Refusing to start {fn}.")
    t0 = time.time()
    try:
        yield
    finally:
        sec = time.time() - t0
        rec = {"fn": fn, "tag": tag, "gpu": gpu, "sec": round(sec, 1),
               "usd": round(sec * RATES[gpu] * OVERHEAD, 4), "t": t0}
        json.dump(rec, open(f"{LEDGER}/{int(t0)}_{fn}_{tag}.json", "w"))
        vol.commit()


def project(sec_per_step, steps, n_runs, gpu="A100-40GB", eval_usd=0.0):
    return sec_per_step * steps * n_runs * RATES[gpu] * OVERHEAD + eval_usd
