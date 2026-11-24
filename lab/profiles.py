"""One switch for pilot vs paper. Each profile gets its OWN Modal app + Volume, results dir, runs file and spec file,
so pilot numbers can never leak into paper results. Edit the dicts to taste; nothing else needs to change."""
import os

_LAMS = [0.0, 0.25, 0.5, 1.0]
_M05, _M15, _M3 = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"

PROFILES = {
    # ~$8-12. Goal: does the pipeline run, do probes have headroom, does lambda bite, what is seed noise?
    "pilot": dict(
        name="pilot", gpu="A10G", cap=15.0, main="m05", models={"m05": (_M05, 8)},
        steps=60, save_every=20, n_prompts=16, group=8, lr=6e-5,
        stage1=[("m05", 0.0, 0), ("m05", 1.0, 0)],
        stage2=[("m05", 0.0, 1), ("m05", 0.5, 0)],      # 2nd lam=0 seed = noise floor
        stage_size=[],
        ckpts=[20, 40, 60], early=20, tier_ckpt="cheap", tier_final="cheap",
        sweep=dict(n_items=8, n=2, dmax=8), samples=dict(n_problems=100, N=8),
        steer_base=[-1.0], steer_rl=[0.5], smoke=["m05"], gate=dict(len_ratio=0.8, acc_drop=0.2)),
    # Template for the real study; costs several x the pilot. Review `plan_cost` in notebook 00 and set `cap` yourself.
    "paper": dict(
        name="paper", gpu="A100-40GB", cap=300.0, main="m15",
        models={"m05": (_M05, 16), "m15": (_M15, 8), "m3": (_M3, 4)},
        steps=300, save_every=50, n_prompts=16, group=8, lr=4e-5,
        stage1=[("m15", 0.0, 0), ("m15", 1.0, 0)],
        stage2=[("m15", l, s) for s in (0, 1, 2) for l in _LAMS if (l, s) not in [(0.0, 0), (1.0, 0)]],
        stage_size=[("m05", l, 0) for l in _LAMS] + [("m3", 0.0, 0), ("m3", 1.0, 0)],
        ckpts=[50, 100, 200, 300], early=50, tier_ckpt="cheap", tier_final="full",
        sweep=dict(n_items=24, n=8, dmax=9), samples=dict(n_problems=1000, N=64),
        steer_base=[-1.0, -0.5, 0.5], steer_rl=[0.5, 1.0], smoke=["m05", "m15", "m3"], gate=dict(len_ratio=0.7, acc_drop=0.15)),
}


def get(name=None):
    """Priority: argument > $TERSE_PROFILE > ./profile.txt (written by notebook 00) > 'pilot'."""
    name = name or os.environ.get("TERSE_PROFILE")
    if not name and os.path.exists("profile.txt"):
        name = open("profile.txt").read().strip()
    return PROFILES[name or "pilot"]


def app_name(p): return f"terse-{p['name']}"
def vol_name(p): return f"terse-{p['name']}-vol"


def all_runs(p):
    return list(p["stage1"]) + list(p["stage2"]) + list(p["stage_size"])


def plan_cost(p, sec_per_step, eval_overhead=0.25):
    """USD projection from measured s/step per model tag (training + ~25% for evals/geometry/sampling)."""
    from lab import cost
    tot, rows = 0.0, {}
    for tag, lam, seed in all_runs(p):
        c = cost.project(sec_per_step[tag], p["steps"], 1, p["gpu"])
        rows[tag] = rows.get(tag, 0.0) + c
    tot = sum(rows.values()) * (1 + eval_overhead)
    return rows, tot
