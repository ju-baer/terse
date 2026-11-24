# TERSE — *Is concision a skill or a character?*

Reasoning-efficiency RL as a character-shaping force with collective consequences, studied in small open models
(Qwen2.5-0.5B/1.5B-Instruct, Apache-2.0) on a ~$100 Modal budget.

## Core question
Efficiency-RL papers report accuracy and length. We ask what else moves when you pay a model to be brief:
1. **H1 Entanglement** – does a length bonus (λ) shift dispositions it never saw in training (abstaining on underdetermined problems, caving to user/peer pushback, calibration), beyond a λ=0 RL control?
2. **H2 One knob?** – is RL's residual-stream shift mostly along a rank-1 "effort" direction, and is that direction entangled with humility/agreeableness? Causal test: steering × RL 2×2.
3. **H3 Collective** – does λ raise the intraclass correlation ρ of i.i.d. samples (shrinking K_eff = N/(1+(N−1)ρ)), and where do depth–breadth token-budget frontiers cross?
4. **H4 Methodology** – can side effects be forecast early (curve extrapolation; step-50 geometry → final drift)?
5. **Harness (stretch)** – with a code-execution harness does penalised effort *relocate* into code rather than disappear?

## Pilot first, then paper
`PROFILE` is set in the first cell of notebook 00 (`"pilot"` or `"paper"`) and the numbers live in `lab/profiles.py`.
Each profile gets its own Modal app, Volume, `results/<profile>/`, `runs_<profile>.json` and `spec_<profile>.json`, so nothing mixes.
- **pilot** (~$8-12): Qwen2.5-0.5B on A10G, 60 steps, λ∈{0,0.5,1}, two λ=0 seeds (noise floor), cheap evals, cap $15.
- **paper** (template, several x the pilot): 0.5B/1.5B/3B, 3 seeds x 4 λ, 300 steps, full-tier finals, 64 samples x 1000 problems. Notebook 00 prints the projected plan cost from measured s/step; set `cap` yourself.
Use the pilot only to fix design choices (λ values, steps, seeds, probe difficulty); write the analysis plan down; then switch profile and rerun notebooks 00-04.

## Layout
```
modal_app.py            Modal app, deployed per profile (notebook 00 does it). Optional overrides: TERSE_GPU, TERSE_CAP
lab/profiles.py         pilot / paper settings
lab/envs.py             procedural verifiable tasks (arith, track, listops, logic) + underdetermined variants
lab/grpo.py             exactly-on-policy Dr.GRPO + correct-only group-relative length bonus, LoRA
lab/probes.py           judge-free probes, persona/effort vectors
lab/collective.py       ICC, K_eff, majority-vote curves, forecasting (numpy)
lab/cost.py             ledger + budget guard;  lab/local.py notebook-side helpers
notebooks/00..04        run in order from VS Code (Python kernel; `pip install modal pandas matplotlib`)
```
Run order: `modal setup` → notebook 00 (deploy, warm-up, **smoke test**) → 01 → 02 (staged, gated) → 03 → 04.

## Status of this code
Tested on CPU: env generators, verifiers, collective statistics, notebook syntax.
**Not executed on GPU** (no GPU/network where it was written). Expect to fix small version/API issues in the smoke test (notebook 00) — that is what it is for.
Library pins: torch 2.6.0, transformers 4.51.3, peft 0.15.2, accelerate 1.6.0.

## Known limitations
Persona vectors are simplified (prompt-contrast, last prompt token, no LLM judge). Only the Qwen2.5 family. Two seeds only for λ∈{0,.5}. 
GRPO has no KL term (LoRA + small LR). Probes are programmatic, so they measure behaviour proxies for "character", not character itself.
