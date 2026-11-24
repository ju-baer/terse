"""Modal app for the TERSE project.  Deploy once:  modal deploy modal_app.py
Notebooks then call the deployed functions with .spawn()/.remote(); jobs survive notebook disconnects.
All artefacts live on the profile's Volume `terse-<profile>-vol` (/logs, /ckpt, /eval, /samples, /geometry, /ledger)."""
import json, os
import modal

from lab import profiles

# Deploy with:  TERSE_PROFILE=pilot modal deploy modal_app.py   (or paper). Optional overrides: TERSE_GPU, TERSE_CAP.
PROFILE = profiles.get(os.environ.get("TERSE_PROFILE", "pilot"))
GPU = os.environ.get("TERSE_GPU", PROFILE["gpu"])
CAP = float(os.environ.get("TERSE_CAP", PROFILE["cap"]))      # soft budget guard in USD, enforced from the ledger

app = modal.App(profiles.app_name(PROFILE))
vol = modal.Volume.from_name(profiles.vol_name(PROFILE), create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0", "numpy")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "TERSE_GPU": GPU, "TERSE_PROFILE": PROFILE["name"]})
    .add_local_python_source("lab")
)
KW = dict(image=image, volumes={"/vol": vol})
GPUKW = dict(gpu=GPU, **KW)


def _spec(cfg):
    return cfg.get("spec") or {"arith": [2, 6], "track": [2, 6], "listops": [2, 6], "logic": [2, 6]}


def _full_spec(cfg):
    s = dict(_spec(cfg))
    s.setdefault("logic", [2, 6])
    return s


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _adapter(cfg, root="/vol"):
    return f"{root}/ckpt/{cfg['name']}/step_{cfg['ckpt']}" if cfg.get("ckpt") else None


@app.function(timeout=1800, **KW)
def warmup(models):
    from huggingface_hub import snapshot_download
    for m in models:
        snapshot_download(m)
    vol.commit()
    return models


@app.function(timeout=3600, **GPUKW)
def smoke(cfg):
    """Tiny end-to-end run: measures s/step so the notebook can project cost before you commit."""
    from lab import cost, grpo
    with cost.track(vol, "smoke", GPU, CAP, cfg.get("name", "smoke")):
        nm = "smoke_" + cfg.get("name", "x")
        grpo.run_grpo({**cfg, "steps": cfg.get("steps", 3), "save_every": 10_000, "name": nm}, vol)
        recs = [json.loads(l) for l in open(f"/vol/logs/{nm}.jsonl")]
    vol.commit()
    return recs


@app.function(timeout=2 * 3600, **GPUKW)
def capability_sweep(cfg):
    import numpy as np
    from lab import cost, envs, modeling
    with cost.track(vol, "capability", GPU, CAP, cfg["model"].split("/")[-1]):
        tok, model = modeling.load(cfg["model"])
        res = {}
        for fam in envs.ALL_FAMILIES:
            res[fam] = {}
            for d in range(1, cfg.get("dmax", 9) + 1):
                items = [envs.make_item(fam, d, i, ns="sweep") for i in range(cfg.get("n_items", 12))]
                prompts = [modeling.chat_text(tok, envs.build_messages(it)) for it in items]
                outs = modeling.generate(model, tok, prompts, n=cfg.get("n", 4), max_new=cfg.get("max_new", 512))
                ok = [[envs.is_correct(it, o["text"]) and not o["truncated"] for o in os_] for it, os_ in zip(items, outs)]
                res[fam][d] = dict(acc=float(np.mean(ok)), len=float(np.mean([o["n_tokens"] for os_ in outs for o in os_])))
        _save(f"/vol/eval/capability_{cfg['model'].split('/')[-1]}.json", res)
    return res


@app.function(timeout=4 * 3600, **GPUKW)
def train_run(cfg):
    from lab import cost, grpo
    with cost.track(vol, "train", GPU, CAP, cfg["name"]):
        return grpo.run_grpo(cfg, vol)


@app.function(timeout=2 * 3600, **GPUKW)
def eval_ckpt(cfg):
    """cfg: name, ckpt (None = base model), model, tier, harness, steer={'trait','alpha'}|None, out_tag"""
    from lab import cost, modeling, probes
    spec = _full_spec(cfg)
    with cost.track(vol, "eval", GPU, CAP, cfg["out_tag"]):
        tok, model = modeling.load(cfg["model"], adapter=_adapter(cfg))
        steer = None
        if cfg.get("steer"):
            with modeling.base_ctx(model):
                vecs = probes.trait_vectors(model, tok, spec)
            L = cfg["steer"].get("layer") or len(modeling.get_layers(model)) // 2
            steer = (L, vecs[cfg["steer"]["trait"]][L], cfg["steer"]["alpha"])
        met, recs = probes.run_battery(model, tok, spec, cfg.get("tier", "cheap"), cfg.get("harness", "cot"),
                                       cfg.get("max_new", 512), steer)
        _save(f"/vol/eval/{cfg['out_tag']}.json", dict(cfg=cfg, metrics=met, records=recs))
    return met


@app.function(timeout=2 * 3600, **GPUKW)
def sample_many(cfg):
    """N samples/problem for each model tag -> /vol/samples/{tag}.jsonl (answers only, no text)."""
    from lab import cost, envs, modeling
    spec = {k: v for k, v in _full_spec(cfg).items() if k in envs.TRAIN_FAMILIES}
    items = envs.make_batch("coll", 0, cfg.get("n_problems", 300), spec)
    truth = [it.answer for it in items]
    _save("/vol/samples/_truth.json", dict(truth=truth, family=[i.family for i in items], d=[i.difficulty for i in items]))
    with cost.track(vol, "sample", GPU, CAP, cfg.get("tag", "many")):
        for m in cfg["models"]:
            tok, model = modeling.load(cfg["model"], adapter=_adapter(m))
            prompts = [modeling.chat_text(tok, envs.build_messages(it, cfg.get("harness", "cot"))) for it in items]
            outs = modeling.generate(model, tok, prompts, n=cfg.get("N", 32), max_new=cfg.get("max_new", 512))
            os.makedirs("/vol/samples", exist_ok=True)
            with open(f"/vol/samples/{m['tag']}.jsonl", "w") as f:
                for p, (it, os_) in enumerate(zip(items, outs)):
                    for j, o in enumerate(os_):
                        a = envs.parse_answer(o["text"], cfg.get("harness", "cot"))
                        f.write(json.dumps(dict(p=p, j=j, ans=a, ok=(a == it.answer and not o["truncated"]),
                                                n=o["n_tokens"], tr=o["truncated"])) + "\n")
            del model
            import torch; torch.cuda.empty_cache()
            vol.commit()
    return [m["tag"] for m in cfg["models"]]


@app.function(timeout=3600, **GPUKW)
def geometry(cfg):
    """Where does RL move the residual stream, measured in units of persona-vector contrast?"""
    import torch
    from lab import cost, modeling, probes
    spec = _full_spec(cfg)
    with cost.track(vol, "geometry", GPU, CAP, cfg.get("tag", "geo")):
        tok, base = modeling.load(cfg["model"])
        vecs = probes.trait_vectors(base, tok, spec)
        h0 = probes.neutral_mean_hidden(base, tok, spec)
        names = list(vecs)
        out = dict(layers=h0.shape[0], traits=names,
                   trait_cos=[[[float(torch.nn.functional.cosine_similarity(vecs[a][l], vecs[b][l], dim=0)) for b in names]
                               for a in names] for l in range(h0.shape[0])],
                   trait_norm={n: [float(vecs[n][l].norm()) for l in range(h0.shape[0])] for n in names}, runs={})
        del base; torch.cuda.empty_cache()
        for m in cfg["runs"]:
            tok, model = modeling.load(cfg["model"], adapter=_adapter(m))
            dA = probes.neutral_mean_hidden(model, tok, spec) - h0
            out["runs"][m["tag"]] = dict(
                delta_norm=[float(dA[l].norm()) for l in range(dA.shape[0])],
                proj={n: [float(dA[l] @ vecs[n][l] / (vecs[n][l] @ vecs[n][l])) for l in range(dA.shape[0])] for n in names},
                cos={n: [float(torch.nn.functional.cosine_similarity(dA[l], vecs[n][l], dim=0)) for l in range(dA.shape[0])] for n in names})
            del model; torch.cuda.empty_cache()
        _save(f"/vol/geometry/{cfg.get('tag', 'geo')}.json", out)
    return out
