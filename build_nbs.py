import json, textwrap
def md(t): return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(t).strip("\n").splitlines(True)}
def code(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":textwrap.dedent(t).strip("\n").splitlines(True)}
def save(name, cells):
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}
    json.dump(nb, open(f"notebooks/{name}.ipynb","w"), indent=1)

HEAD = code('''
import sys, json, os, glob, time; sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from lab import profiles, local, envs, collective, cost
P = profiles.get(); MAIN = P["main"]; HF = {t: v[0] for t, v in P["models"].items()}; FINAL = P["steps"]
SPEC = f"spec_{P['name']}.json"
print("PROFILE:", P["name"], "| GPU:", P["gpu"], "| cap $", P["cap"], "| main model:", HF[MAIN], "| steps:", FINAL)
''')

save("00_setup_budget", [
md('''
# 00 · Profile, setup, budget, smoke test
**TERSE — *Is concision a skill or a character?***

One switch below selects a **profile**. `pilot` (~$8–12, 0.5B, A10G) answers: does the pipeline run, do the probes have headroom, does λ bite, what is the seed-noise floor?
`paper` is the template for the real study (several × the cost). Each profile has its **own Modal app, Volume, results folder, runs file and spec file**, so pilot numbers cannot leak into paper results.
Edit `lab/profiles.py` to change seeds / steps / models / tiers — nothing else needs touching.

Workflow: pilot → decide λ values, steps, seeds, probe difficulty → **write down the analysis plan** → switch to `paper`, redeploy (fresh Volume = fresh budget count).
Pilot runs are for design decisions only; do not report them as results.
'''),
code('''
PROFILE = "pilot"            # <- "pilot" or "paper"; re-run the whole notebook after changing
open("profile.txt", "w").write(PROFILE)
%pip install -q modal numpy pandas matplotlib
'''),
md('**One-time auth (terminal):** `modal setup`. If your Modal plan offers a workspace spend limit, set it too — the ledger guard is a soft limit (parallel jobs each check before starting; killed jobs may not write a ledger row).'),
code('''
import sys; sys.path.insert(0, "..")
import modal; print("modal", modal.__version__)
'''),
code('!cd .. && TERSE_PROFILE={PROFILE} modal deploy modal_app.py'),
HEAD,
code('local.fn("warmup").remote([HF[t] for t in P["models"]])   # cache weights on this profile\'s Volume'),
md('### Smoke test: 3 GRPO steps per model → seconds/step → projected cost of the *whole plan*'),
code('''
sps = {}
for tag in P["smoke"]:
    hf, mb = P["models"][tag]
    recs = local.fn("smoke").remote(dict(name=tag, model=hf, steps=3, micro_bs=mb, lam=0.5, lr=P["lr"]))
    sps[tag] = sum(r["sec"] for r in recs[1:]) / max(1, len(recs) - 1)          # skip step 0 (CUDA warm-up)
    print(tag, "s/step ≈", round(sps[tag], 1), "| generation share ≈", round(sum(r["t_gen"] for r in recs) / sum(r["sec"] for r in recs), 2), "| last:", recs[-1])
json.dump(sps, open(f"sec_per_step_{P['name']}.json", "w"))
'''),
code('''
rows, tot = profiles.plan_cost(P, sps)
print("training cost by model ($):", {k: round(v, 1) for k, v in rows.items()})
print(f"PLAN TOTAL (train + ~25% evals/geometry/sampling) ≈ ${tot:.2f}  vs cap ${P['cap']:.0f}")
if tot > P["cap"]: print("!! plan exceeds cap — cut seeds/steps/models in lab/profiles.py before launching")
df, spent = local.ledger(); print(f"SPENT SO FAR: ${spent:.2f}")
'''),
md('**Decision rule.** Proceed only if the smoke test ran and the projected plan fits under the cap. For `paper`, if 1.5B s/step is > 90, cut `steps` or move to a cheaper-per-token setup (e.g. vLLM rollouts) before spending.'),
])

save("01_environments_baselines", [
md('''
# 01 · Open procedural environments, difficulty calibration, base-model battery
**Topic 3 (open RL environments).** Each item is a pure function of `(namespace, family, difficulty, index)`: unlimited data, disjoint splits by construction,
programmatic verifiers, a difficulty knob, and underdetermined variants (answer = `UNDETERMINED`) used only to probe dispositions.
RL trains on `arith / track / listops`; `logic` and all underdetermined / pushback / confidence probes are held out.
'''),
HEAD,
code('''
for fam in envs.ALL_FAMILIES:
    it = envs.make_item(fam, 3, 0, "demo"); print(f"--- {fam} (answer {it.answer})\\n{it.question}\\n")
u = envs.make_item("track", 3, 1, "demo", und=True); print("--- underdetermined track (answer", u.answer, ")\\n" + u.question)
its = [envs.make_item(f, d, i, "chk") for f in envs.ALL_FAMILIES for d in (1, 4, 8) for i in range(50)]
print("oracle:", np.mean([envs.is_correct(i, f"<answer>{i.answer}</answer>") for i in its]), "| wrong-answer:", np.mean([envs.is_correct(i, f"<answer>{i.wrong}</answer>") for i in its]))
'''),
md('### Capability sweep of the main model → *learnability band* (pass-rate 0.15–0.85) per family'),
code('''
sw = P["sweep"]
sweep = local.fn("capability_sweep").remote(dict(model=HF[MAIN], n_items=sw["n_items"], n=sw["n"], dmax=sw["dmax"]))
tab = pd.DataFrame({f: {int(d): v["acc"] for d, v in sweep[f].items()} for f in sweep})
fig, ax = plt.subplots(figsize=(5, 3.5)); im = ax.imshow(tab.T.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index); ax.set_yticks(range(len(tab.columns))); ax.set_yticklabels(tab.columns)
ax.set_xlabel("difficulty"); plt.colorbar(im, label="base pass@1"); plt.show(); tab.round(2)
'''),
code('''
def band(fam, lo_p=0.15, hi_p=0.85):
    ds = [int(d) for d in tab.index if lo_p <= tab.loc[d, fam] <= hi_p]
    return [min(ds), max(ds)] if ds else [2, 5]
spec = {f: band(f) for f in envs.ALL_FAMILIES}; print(spec); json.dump(spec, open(SPEC, "w"))
# Training uses only TRAIN_FAMILIES; 'logic' ranges are for held-out evaluation.
'''),
md('### Base-model behavioural battery: accuracy, abstention on underdetermined problems, pushback sycophancy, calibration'),
code('''
tag = f"ev_{MAIN}_base"
local.launch("eval_ckpt", dict(name="base", ckpt=None, model=HF[MAIN], tier=P["tier_final"], spec=spec, out_tag=tag), key=tag)
print(local.wait(tag)); local.pull("/eval"); b = json.load(open(local.find(f"{tag}.json")[0]))["metrics"]; pd.Series(b).round(3)
'''),
md('''
**Headroom check (the main pilot gate).** `abstain_recall` should be neither ~0 nor ~1; `changed_neutral` (spontaneous revision) should be small relative to `flip_user`/`flip_peer`;
`conf_parse` should be high. If a probe has no headroom, change difficulty or phrasing *now* — it is the cheapest fix in the whole project.
'''),
])

save("02_rl_sweep", [
md('''
# 02 · GRPO with a controllable efficiency pressure λ
Reward = 1[correct] + λ·(1 − normalised length among the *correct* samples of the group); wrong or truncated = 0.
Dr.GRPO-style mean-centred advantages (λ is in reward units), exactly on-policy, LoRA r=32. **Staged with a gate** so a dead hypothesis costs little.
Stage lists (`stage1`, `stage2`, `stage_size`) come from the profile.
'''),
HEAD,
code('''
spec = json.load(open(SPEC)); train_spec = {f: spec[f] for f in envs.TRAIN_FAMILIES}
sps = json.load(open(f"sec_per_step_{P['name']}.json")); print(sps, train_spec)
def mk(tag, lam, seed, **kw):
    hf, mb = P["models"][tag]
    return dict(name=f"{tag}_lam{lam}_s{seed}", model=hf, lam=lam, seed=seed, steps=P["steps"], save_every=P["save_every"], n_prompts=P["n_prompts"],
                group=P["group"], lr=P["lr"], micro_bs=mb, spec=train_spec, **kw)
def logdf(name):
    fs = local.find(f"{name}.jsonl", must="logs"); return pd.read_json(fs[0], lines=True) if fs else None
'''),
md('## Stage 1 — the poles'),
code('''
for tag, lam, seed in P["stage1"]: local.launch("train_run", mk(tag, lam, seed))
print(local.status(MAIN))
'''),
code('''
local.pull("/logs"); print(local.status(MAIN))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
for tag, lam, seed in P["stage1"]:
    d = logdf(f"{tag}_lam{lam}_s{seed}")
    if d is None: continue
    for a, k in zip(ax, ["acc", "len", "trunc"]): a.plot(d.step, d[k].rolling(10, min_periods=1).mean(), label=f"λ={lam} s{seed}"); a.set_title(k)
ax[0].legend(); plt.show()
'''),
code('''
def tail(name, k=None):
    d = logdf(name); k = k or max(5, len(d) // 4); return d.tail(k)[["acc", "len", "len_correct", "trunc"]].mean()
lo, hi = P["stage1"][0], P["stage1"][-1]
a, b = tail(f"{lo[0]}_lam{lo[1]}_s{lo[2]}"), tail(f"{hi[0]}_lam{hi[1]}_s{hi[2]}"); print(pd.DataFrame({"lam_lo": a, "lam_hi": b}).round(3))
g = P["gate"]; GATE = bool(b.len < g["len_ratio"] * a.len and b.acc > a.acc - g["acc_drop"])
print(f"GATE 1 (length ×<{g['len_ratio']} and accuracy drop <{g['acc_drop']}):", GATE)
'''),
md('**If GATE 1 fails:** raise `lr` or the top λ in `lab/profiles.py`, redeploy, rerun Stage 1 only. Do not continue.'),
code('''
if GATE:
    for tag, lam, seed in P["stage2"]: local.launch("train_run", mk(tag, lam, seed))
print(local.status(MAIN))
'''),
md('''
**Pilot noise floor.** In the pilot, `stage2` contains a second λ=0 seed. The gap between the two λ=0 runs below is your seed noise — if it is as large as the λ effect,
the paper needs more seeds/steps, not more λ values.
'''),
code('''
local.wait(MAIN, every=60); local.pull("/logs")
runs0 = [r for r in local.status(MAIN)]; rows = {r: tail(r) for r in runs0 if logdf(r) is not None}
print(pd.DataFrame(rows).T.round(3).sort_index())
'''),
md('## Size axis (paper profile only: other model sizes)'),
code('''
if GATE and P["stage_size"]:
    for tag, lam, seed in P["stage_size"]: local.launch("train_run", mk(tag, lam, seed))
    print(local.wait("m", every=120)); local.pull("/logs")
df, spent = local.ledger(); print(df.groupby("fn").usd.sum().round(2), f"\\nSPENT: ${spent:.2f} of ${P['cap']:.0f}")
'''),
])

save("03_dispositions_geometry", [
md('''
# 03 · Does efficiency pressure change *who the model is*?  And is it one knob?
**Topic 2 (character) × Topic 4 (reasoning efficiency).**
*H1.* λ>0 moves non-targeted dispositions (abstention on underdetermined problems, caving to pushback, calibration) beyond λ=0 RL — dose-dependently.
*H2.* RL's residual shift projects mostly onto an "effort" direction that is not orthogonal to humility/agreeableness. Tests: geometry + steering×RL 2×2.
'''),
HEAD,
code('''
spec = json.load(open(SPEC))
runs = sorted(k for k in json.load(open(local.runs_file())) if k.startswith(MAIN + "_lam")); print(runs)
'''),
md('### Checkpoint batteries (tiers from the profile)'),
code('''
for r in runs:
    for s in P["ckpts"]:
        t = f"ev_{r}_{s}"
        local.launch("eval_ckpt", dict(name=r, ckpt=s, model=HF[MAIN], tier=P["tier_final"] if s == FINAL else P["tier_ckpt"], spec=spec, out_tag=t), key=t)
print(local.wait(f"ev_{MAIN}", every=90)); local.pull("/eval")
'''),
code('''
rows = []
for f in local.find(f"ev_{MAIN}_*.json"):
    j = json.load(open(f)); n = os.path.basename(f)[3:-5]
    if n == f"{MAIN}_base": run, step, lam, seed = "base", 0, 0.0, -1
    else:
        run, step = n.rsplit("_", 1); step = int(step); lam = float(run.split("lam")[1].split("_")[0]); seed = int(run.split("_s")[-1])
    rows.append(dict(run=run, lam=lam, seed=seed, step=step, **j["metrics"]))
E = pd.DataFrame(rows).sort_values(["lam", "seed", "step"]); E.to_csv(f"evals_{P['name']}.csv", index=False)
fin = E[(E.step == FINAL) | (E.run == "base")]
cols = [c for c in ["acc_id", "acc_ood", "len_id", "abstain_recall", "abstain_false", "fabricate", "flip_user", "flip_peer", "changed_neutral", "calib_auroc", "calib_ece"] if c in fin]
print(fin.groupby(["lam", "seed"])[cols].mean().round(3))
'''),
code('''
def boot(recs, key, filt=lambda r: True, B=2000, seed=0):
    v = np.array([float(r[key]) for r in recs if filt(r)]); rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v)).mean() for _ in range(B)]; return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
panels = [("abst", "answered_und", lambda r: r["und"], "abstain recall (UND items)"),
          ("push", "flip", lambda r: r["kind"] == "user", "cave to user pushback"),
          ("push", "flip", lambda r: r["kind"] == "peer", "cave to peer answer"),
          ("acc", "correct", lambda r: r["group"] == "id", "accuracy (in-dist)")]
def final_recs(lam):
    recs = []
    for f in local.find(f"ev_{MAIN}_lam{lam}_s*_{FINAL}.json"): recs += json.load(open(f))["records"]
    return recs
base_recs = json.load(open(local.find(f"ev_{MAIN}_base.json")[0]))["records"]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for a, (probe, key, filt, title) in zip(ax, panels):
    m, lo, hi = boot([r for r in base_recs if r["probe"] == probe], key, filt); a.axhline(m, ls="--", color="gray"); a.axhspan(lo, hi, color="gray", alpha=.15)
    for lam in sorted(fin[fin.run != "base"].lam.unique()):
        recs = [r for r in final_recs(lam) if r["probe"] == probe]
        if recs: m, lo, hi = boot(recs, key, filt); a.errorbar(lam, m, yerr=[[m - lo], [hi - m]], fmt="o", capsize=3, color="C0")
    a.set_title(title + "  (dashed = base)", fontsize=8); a.set_xlabel("λ")
plt.tight_layout(); plt.show()
'''),
md('### Geometry: where does RL move the residual stream, in units of persona contrast?'),
code('''
runs_geo = [dict(tag=f"{r}_{s}", name=r, ckpt=s) for r in runs for s in (P["early"], FINAL)]
geo = local.fn("geometry").remote(dict(model=HF[MAIN], spec=spec, runs=runs_geo, tag=f"geo_{MAIN}")); L = geo["layers"]; mid = L // 2
print("trait–trait cosine at mid layer:", geo["traits"]); print(np.array(geo["trait_cos"][mid]).round(2))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
for tag, v in geo["runs"].items():
    if not tag.endswith(f"_{FINAL}"): continue
    for a, t in zip(ax, geo["traits"]): a.plot(v["proj"][t], label=tag)
for a, t in zip(ax, geo["traits"]): a.set_title(f"RL shift on {t} (1 = full persona contrast)", fontsize=8); a.set_xlabel("layer")
ax[0].legend(fontsize=6); plt.show()
'''),
code('''
pts = []
b0 = float(fin[fin.run == "base"].abstain_recall.iloc[0])
for tag, v in geo["runs"].items():
    if not tag.endswith(f"_{FINAL}"): continue
    e = fin[fin.run == tag[:-len(str(FINAL)) - 1]]
    if len(e): pts.append((v["proj"]["humility"][mid], float(e.abstain_recall.iloc[0]) - b0, v["proj"]["effort"][mid]))
Pt = np.array(pts); print("proj_humility, Δabstain, proj_effort\\n", Pt.round(3))
plt.scatter(Pt[:, 0], Pt[:, 1]); plt.xlabel("projection of RL shift on humility vector"); plt.ylabel("Δ abstain recall vs base"); plt.show()
'''),
md('''
### Steering × RL dissociation (the central causal test)
Terseness reproduces the drift ⇒ the entanglement lives in the effort axis. Adding effort back restores abstention ⇒ the effort axis **mediates** it. Neither ⇒ RL changed other things.
'''),
code('''
hi = P["stage1"][-1]; r1 = f"{hi[0]}_lam{hi[1]}_s{hi[2]}"
for al in P["steer_base"]:
    t = f"st_base_{al}"; local.launch("eval_ckpt", dict(name="base", ckpt=None, model=HF[MAIN], tier=P["tier_final"], spec=spec, steer=dict(trait="effort", alpha=al), out_tag=t), key=t)
for al in P["steer_rl"]:
    t = f"st_{r1}_{al}"; local.launch("eval_ckpt", dict(name=r1, ckpt=FINAL, model=HF[MAIN], tier=P["tier_final"], spec=spec, steer=dict(trait="effort", alpha=al), out_tag=t), key=t)
print(local.wait("st_", every=90)); local.pull("/eval")
S = {os.path.basename(f)[3:-5]: json.load(open(f))["metrics"] for f in local.find("st_*.json")}
S["base (no steer)"] = json.load(open(local.find(f"ev_{MAIN}_base.json")[0]))["metrics"]; S[f"{r1} (no steer)"] = json.load(open(local.find(f"ev_{r1}_{FINAL}.json")[0]))["metrics"]
T = pd.DataFrame(S).T; T[[c for c in ["acc_id", "len_id", "abstain_recall", "fabricate", "flip_user", "flip_peer", "calib_auroc"] if c in T]].round(3)
'''),
])

save("04_collective_forecast_report", [
md('''
# 04 · Collective consequences, forecasting, final report
**Topic 6 × Topic 5.** *H3:* λ raises the intraclass correlation ρ of correctness across i.i.d. samples, so K_eff = N/(1+(N−1)ρ) shrinks — but efficient models give more votes per token.
Where do depth–breadth token-budget frontiers cross? *H4:* side effects are forecastable early (curve extrapolation; early geometry → final drift).
'''),
HEAD,
code('''
spec = json.load(open(SPEC))
runs = sorted(k for k in json.load(open(local.runs_file())) if k.startswith(MAIN + "_lam") and k.endswith("_s0"))
models = [dict(tag="base", name="base", ckpt=None)] + [dict(tag=r, name=r, ckpt=FINAL) for r in runs]
sm = P["samples"]
local.launch("sample_many", dict(model=HF[MAIN], spec=spec, models=models, n_problems=sm["n_problems"], N=sm["N"], tag="coll"), key="samples_coll")
print(local.wait("samples", every=90)); local.pull("/samples")
'''),
code('''
T = json.load(open(local.find("_truth.json")[0])); truth = T["truth"]; NP, N = len(truth), sm["N"]
def load(tag):
    d = pd.read_json(local.find(f"{tag}.jsonl", must="samples")[0], lines=True)
    C = np.zeros((NP, N), int); A = [[None] * N for _ in range(NP)]; Ln = np.zeros((NP, N))
    for r in d.itertuples(): C[r.p, r.j] = r.ok; A[r.p][r.j] = r.ans; Ln[r.p, r.j] = r.n
    return C, A, Ln
rows, data = [], {}
for m in models:
    C, A, Ln = load(m["tag"]); data[m["tag"]] = (C, A, Ln); rho = collective.icc_binary(C)
    rows.append(dict(model=m["tag"], acc=C.mean(), mean_len=Ln.mean(), ICC=rho, K_eff=collective.k_eff(N, rho),
                     wrong_mode_share=collective.wrong_mode_concentration(A, truth), pass_at_4=collective.pass_at_k(C, 4), maj=collective.maj_curve(A, truth, [N - 1], R=30)[0]))
pd.DataFrame(rows).round(3)
'''),
code('''
ns = [n for n in (1, 2, 3, 5, 9, 15, 23, 31, 47, 63) if n < N]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for tag, (C, A, Ln) in data.items():
    y = collective.maj_curve(A, truth, ns, R=40); ax[0].plot(ns, y, "o-", label=tag); ax[1].plot(np.array(ns) * Ln.mean(), y, "o-")
ax[0].set_xlabel("n votes"); ax[0].set_ylabel("majority-vote accuracy"); ax[1].set_xscale("log"); ax[1].set_xlabel("total tokens per problem (n × mean length)")
ax[1].set_title("depth–breadth frontier: look for crossings"); ax[0].legend(fontsize=6); plt.tight_layout(); plt.show()
'''),
md('### Forecasting (H4a): extrapolate logged metrics from the first 40% of training vs a persistence baseline'),
code('''
local.pull("/logs"); out = []; bin_ = max(1, FINAL // 20)
for r in runs:
    d = pd.read_json(local.find(f"{r}.jsonl", must="logs")[0], lines=True); dd = d.groupby(d.step // bin_).mean(numeric_only=True)
    for k in ("len", "acc", "trunc"):
        out.append(dict(run=r, metric=k, **collective.forecast_final(dd.index.values * bin_, dd[k].values, frac=0.4)))
F = pd.DataFrame(out); print(F.groupby("metric")[["err_fit", "err_persist"]].mean().round(4))
'''),
md('### Forecasting (H4b): early geometry → final drift (small n — report honestly)'),
code('''
geo = json.load(open(local.find(f"geo_{MAIN}.json")[0])); mid = geo["layers"] // 2; E = pd.read_csv(f"evals_{P['name']}.csv"); base = E[E.run == "base"].iloc[0]
pts = []
for r in runs:
    e = E[(E.run == r) & (E.step == FINAL)]
    if len(e): pts.append(dict(run=r, early_proj_hum=geo["runs"][f"{r}_{P['early']}"]["proj"]["humility"][mid], early_proj_eff=geo["runs"][f"{r}_{P['early']}"]["proj"]["effort"][mid],
                               d_abstain=float(e.abstain_recall.iloc[0] - base.abstain_recall), d_flip_user=float(e.flip_user.iloc[0] - base.flip_user)))
Q = pd.DataFrame(pts); print(Q.round(3)); print(Q.drop(columns="run").corr(method="spearman").round(2))
'''),
md('''
### Report template
1. **Pareto:** length–accuracy frontier per λ; training-free `effort` steering overlaid.
2. **H1:** Δ disposition vs λ with bootstrap CIs, λ=0 RL control vs λ>0. Report nulls as nulls.
3. **H2:** projection table + steering×RL 2×2; state which outcome (mediates / reproduces / neither).
4. **H3:** ICC, K_eff, wrong-mode share vs λ; token-budget crossing point if any.
5. **H4:** forecast error vs persistence; geometry→drift Spearman (small n).
6. **Budget:** ledger below. Limitations: Qwen2.5 family, simplified persona vectors, programmatic probes.
'''),
code('df, spent = local.ledger(); print(df.groupby("fn").usd.sum().round(2)); print(f"TOTAL SPENT ${spent:.2f} of ${P[\'cap\']:.0f}")'),
])
print("ok")
