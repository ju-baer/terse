"""Notebook-side helpers (run on your laptop / VS Code kernel, no GPU)."""
import json, os, subprocess, time
import modal

from lab import profiles


def _p():
    return profiles.get()



def fn(name):
    return modal.Function.from_name(profiles.app_name(_p()), name)


def runs_file():
    return f"runs_{_p()['name']}.json"


def _load():
    return json.load(open(runs_file())) if os.path.exists(runs_file()) else {}


def launch(fn_name, cfg, key=None):
    call = fn(fn_name).spawn(cfg)
    runs = _load()
    runs[key or cfg.get("out_tag") or cfg.get("name") or fn_name] = dict(call_id=call.object_id, fn=fn_name, t=time.time())
    json.dump(runs, open(runs_file(), "w"), indent=1)
    return call.object_id


def status(prefix=""):
    rows = {}
    for k, v in _load().items():
        if not k.startswith(prefix):
            continue
        try:
            modal.FunctionCall.from_id(v["call_id"]).get(timeout=0)
            rows[k] = "done"
        except TimeoutError:
            rows[k] = f"running ({(time.time() - v['t']) / 60:.0f} min)"
        except Exception as e:
            rows[k] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
    return rows


def wait(prefix="", every=60):
    while True:
        s = status(prefix)
        if not any(v.startswith("running") for v in s.values()):
            return s
        time.sleep(every)


def pull(vol_path):
    """Download a Volume directory to results/<dir>. (CLI may nest one level; find() copes.)"""
    local = os.path.join("results", _p()["name"], vol_path.strip("/"))
    os.makedirs(local, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", profiles.vol_name(_p()), vol_path, local], check=True)


def ledger():
    import pandas as pd
    vol = modal.Volume.from_name(profiles.vol_name(_p()))
    rows = []
    try:
        for e in vol.listdir("/ledger"):
            rows.append(json.loads(b"".join(vol.read_file(e.path))))
    except Exception:
        pass
    df = pd.DataFrame(rows)
    return df, (float(df.usd.sum()) if len(df) else 0.0)


def find(pattern, must="", root=None):
    """Glob under results/; `must` filters by path substring (e.g. 'logs' vs 'samples' share filenames)."""
    import glob
    root = root or os.path.join("results", _p()["name"])
    return [f for f in sorted(glob.glob(f"{root}/**/{pattern}", recursive=True)) if must in f]
