"""Minimal, exactly-on-policy GRPO (Dr.GRPO-style: mean-centred advantages, constant length normaliser).

One optimiser step per rollout batch => no importance ratio / clipping needed. LoRA keeps it cheap.
Length shaping: correct samples get a bonus lam * (1 - normalised length) *within their group*.
Wrong / truncated samples get 0, so a long-correct answer is never worse than a wrong one.
"""
import json, os, re, time
import numpy as np

DEFAULT = dict(name="run", model="Qwen/Qwen2.5-1.5B-Instruct", steps=200, n_prompts=16, group=8, max_new=512,
               lr=4e-5, lora_r=32, micro_bs=8, lam=0.0, harness="cot", length_scope="all", save_every=25,
               seed=0, spec={"arith": [2, 6], "track": [2, 6], "listops": [2, 6]})


def shaped_rewards(correct, lens, lam):
    r = [float(c) for c in correct]
    idx = [i for i, c in enumerate(correct) if c]
    if lam > 0 and len(idx) >= 2:
        Ls = [lens[i] for i in idx]
        lo, hi = min(Ls), max(Ls)
        if hi > lo:
            for i in idx:
                r[i] += lam * (1.0 - (lens[i] - lo) / (hi - lo))
    return r


def effort_len(tok, text, n_tokens, harness, scope):
    if harness == "pal" and scope == "prose":
        return len(tok(re.sub(r"```.*?```", "", text, flags=re.S)).input_ids)
    return n_tokens


def run_grpo(cfg, vol=None, root="/vol"):
    import torch
    from lab import envs, modeling
    c = {**DEFAULT, **cfg}
    torch.manual_seed(c["seed"])
    os.makedirs(f"{root}/logs", exist_ok=True)
    log_path = f"{root}/logs/{c['name']}.jsonl"
    tok, model = modeling.load(c["model"], lora={"r": c["lora_r"]})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=0.0)
    P, G = c["n_prompts"], c["group"]
    B = P * G
    log = open(log_path, "a")
    for step in range(c["steps"]):
        t0 = time.time()
        items = envs.make_batch(f"train{c['seed']}", step, P, c["spec"])
        prompts = [modeling.chat_text(tok, envs.build_messages(it, c["harness"])) for it in items]
        ro = modeling.rollout(model, tok, [p for p in prompts for _ in range(G)], c["max_new"])
        t_gen = time.time() - t0
        rep_items = [items[k // G] for k in range(B)]
        ok = envs.is_correct_batch(rep_items, ro["texts"], c["harness"])
        correct = [bool(o) and not t for o, t in zip(ok, ro["trunc"])]
        eff = [effort_len(tok, tx, L, c["harness"], c["length_scope"]) for tx, L in zip(ro["texts"], ro["lens"])]
        rewards, adv = [], []
        for g in range(P):
            sl = slice(g * G, (g + 1) * G)
            r = shaped_rewards(correct[sl], eff[sl], c["lam"])
            m = sum(r) / G
            rewards += r
            adv += [x - m for x in r]
        adv_t = torch.tensor(adv, device="cuda")
        nz = [k for k in range(B) if abs(adv[k]) > 1e-8]
        rec = dict(step=step, acc=float(np.mean(correct)), reward=float(np.mean(rewards)),
                   len=float(np.mean(ro["lens"])), len_correct=float(np.mean([l for l, o in zip(ro["lens"], correct) if o] or [0])),
                   trunc=float(np.mean(ro["trunc"])), n_update=len(nz), t_gen=round(t_gen, 1))
        if nz:
            model.train()
            opt.zero_grad(set_to_none=True)
            norm = B * c["max_new"]
            for s in range(0, len(nz), c["micro_bs"]):
                mb = torch.tensor(nz[s:s + c["micro_bs"]], device="cuda")
                lp = modeling.token_logps(model, ro["prompt_ids"][mb], ro["prompt_mask"][mb], ro["comp"][mb], ro["comp_mask"][mb])
                loss = -(adv_t[mb][:, None] * lp * ro["comp_mask"][mb]).sum() / norm
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        rec["sec"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % c["save_every"] == 0 or step + 1 == c["steps"]:
            model.save_pretrained(f"{root}/ckpt/{c['name']}/step_{step + 1}")
            if vol is not None:
                vol.commit()
    log.close()
    return {"name": c["name"], "final": rec}
