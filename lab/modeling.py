"""Thin HF wrappers: load, batched sampling, rollouts for GRPO, activations, steering."""
import contextlib
import torch
import torch.nn.functional as F

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(model_name, adapter=None, lora=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    if lora:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=2 * lora["r"], lora_dropout=0.0,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    elif adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.config.use_cache = True
    return tok, model


def eos_ids(tok):
    return [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]


def chat_text(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _first_eos_mask(comp, eos):
    is_eos = torch.zeros_like(comp, dtype=torch.bool)
    for e in eos:
        is_eos |= comp == e
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((comp.shape[0],), comp.shape[1], device=comp.device))
    idx = torch.arange(comp.shape[1], device=comp.device)[None, :]
    mask = idx <= first[:, None]                      # include the eos token itself
    truncated = ~is_eos.any(1)
    lens = mask.sum(1)
    return mask.long(), truncated, lens


@torch.no_grad()
def _gen_chunk(model, tok, prompts, max_new, temperature, top_p):
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    g = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                       repetition_penalty=1.0, max_new_tokens=max_new,
                       eos_token_id=eos_ids(tok), pad_token_id=tok.pad_token_id)
    comp = g[:, enc.input_ids.shape[1]:]
    mask, trunc, lens = _first_eos_mask(comp, eos_ids(tok))
    return enc, comp, mask, trunc, lens


def generate(model, tok, prompts, n=1, max_new=512, temperature=1.0, top_p=1.0, bs=256):
    """Returns list over prompts of list over n samples of dict(text, n_tokens, truncated)."""
    flat = [(i, p) for i, p in enumerate(prompts) for _ in range(n)]
    out = [[] for _ in prompts]
    for s in range(0, len(flat), bs):
        chunk = flat[s:s + bs]
        enc, comp, mask, trunc, lens = _gen_chunk(model, tok, [p for _, p in chunk], max_new, temperature, top_p)
        for (i, _), row, L, t in zip(chunk, comp, lens.tolist(), trunc.tolist()):
            out[i].append({"text": tok.decode(row[:L], skip_special_tokens=True), "n_tokens": int(L), "truncated": bool(t)})
    return out


def rollout(model, tok, prompts, max_new, temperature=1.0):
    """One big on-policy batch for GRPO; keeps tensors for the update."""
    enc, comp, mask, trunc, lens = _gen_chunk(model, tok, prompts, max_new, temperature, 1.0)
    texts = [tok.decode(r[:L], skip_special_tokens=True) for r, L in zip(comp, lens.tolist())]
    return dict(prompt_ids=enc.input_ids, prompt_mask=enc.attention_mask, comp=comp, comp_mask=mask,
                trunc=trunc.tolist(), lens=lens.tolist(), texts=texts)


def token_logps(model, prompt_ids, prompt_mask, comp, comp_mask):
    """Per-token log-probs of completion tokens under the current policy (with grad)."""
    T = comp.shape[1]
    ids = torch.cat([prompt_ids, comp], 1)
    attn = torch.cat([prompt_mask, comp_mask], 1)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=T + 1).logits[:, :-1, :]
    lp = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), comp.reshape(-1), reduction="none")
    return lp.view(comp.shape)


# ------------------------------------------------------------------ activations & steering
def get_layers(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


@torch.no_grad()
def last_token_hidden(model, tok, prompts, bs=64):
    """(N, n_layers, d) residual stream at the final prompt token, after each layer."""
    model.eval()
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        outs.append(torch.stack([h[:, -1, :].float().cpu() for h in hs[1:]], 1))
    return torch.cat(outs)


@contextlib.contextmanager
def steering(model, layer, vec, alpha):
    v = (alpha * vec).to(model.device, model.dtype)

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + v,) + tuple(out[1:])
        return out + v

    h = get_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def base_ctx(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
