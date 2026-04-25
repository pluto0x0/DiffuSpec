#!/usr/bin/env python3
"""
Naive speculative decoding: DLM draft → AR greedy verify.

Each step:
  1. DLM runs one forward pass on [prefix | MASK*k] → k argmax draft tokens
  2. AR runs one parallel forward pass → accepts the longest matching prefix
     (on the first mismatch, takes the AR token as replacement;
      if all k match, appends one free bonus token)

Usage:
    python scripts/naive_spec_dec.py \
        --prompt "Explain quantum computing." \
        --target  Qwen/Qwen3-8B \
        --drafter dream-org/dream-v0-instruct-7b \
        --draft-len 5 \
        --max-new-tokens 200 \
        --device cuda
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel, AutoTokenizer


# ── Model loading ────────────────────────────────────────────────────────────

def load_models(target_name, drafter_name, device, dtype=torch.bfloat16):
    print(f"Loading target:  {target_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=dtype, device_map=device, trust_remote_code=True
    ).eval()

    print(f"Loading drafter: {drafter_name}")
    try:
        drafter = AutoModelForMaskedLM.from_pretrained(
            drafter_name, torch_dtype=dtype, device_map=device, trust_remote_code=True
        ).eval()
    except Exception:
        drafter = AutoModel.from_pretrained(
            drafter_name, torch_dtype=dtype, device_map=device, trust_remote_code=True
        ).eval()

    return tokenizer, target, drafter


# ── Core speculative decoding steps ─────────────────────────────────────────

@torch.no_grad()
def dlm_draft(drafter, prefix_ids: torch.Tensor, k: int, mask_id: int, vocab_size: int):
    """
    DLM forward pass on [prefix | MASK*k].
    Dream-7B is causal: logits[i] predicts position i+1, so the first draft
    position is predicted by logits[prefix_len - 1].
    Returns k argmax token IDs.
    """
    device = prefix_ids.device
    masks = torch.full((k,), mask_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prefix_ids, masks]).unsqueeze(0)        # [1, prefix+k]

    out = drafter(input_ids=input_ids)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0]  # [L, V]

    plen = prefix_ids.shape[0]
    draft_logits = logits[plen - 1 : plen - 1 + k].clone()         # [k, V]
    draft_logits[:, vocab_size:] = float("-inf")                    # mask drafter-only tokens
    return draft_logits.argmax(dim=-1)                              # [k]


@torch.no_grad()
def ar_verify(target, prefix_ids: torch.Tensor, draft_ids: torch.Tensor, stop_ids: set):
    """
    Single parallel AR forward pass over [prefix | draft].
    Accept greedily: keep draft[i] while target agrees; take target's token on
    the first mismatch. Append one bonus token when all k draft tokens match.

    Returns:
        accepted_ids    – tokens to append to context
        n_draft_accepted – number of draft tokens the AR model agreed with
        hit_eos         – True if a stop token is in accepted_ids
    """
    full_ids = torch.cat([prefix_ids, draft_ids]).unsqueeze(0)      # [1, L]
    out = target(input_ids=full_ids, use_cache=False)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0]  # [L, V]

    plen = prefix_ids.shape[0]
    k = draft_ids.shape[0]
    ar_greedy = logits[plen - 1 : plen - 1 + k].argmax(dim=-1)     # [k]

    accepted = []
    n_draft_accepted = 0
    for i in range(k):
        t = ar_greedy[i].item()
        if t == draft_ids[i].item():
            accepted.append(ar_greedy[i])
            n_draft_accepted += 1
            if t in stop_ids:
                return torch.stack(accepted), n_draft_accepted, True
        else:
            accepted.append(ar_greedy[i])                           # replacement token
            return torch.stack(accepted), n_draft_accepted, t in stop_ids

    # All k matched → free bonus token from the very next position
    bonus = logits[plen - 1 + k].argmax()
    accepted.append(bonus)
    return torch.stack(accepted), n_draft_accepted, bonus.item() in stop_ids


# ── Generation loop ──────────────────────────────────────────────────────────

def generate(target, drafter, tokenizer, prompt_ids, draft_len, max_new_tokens, device):
    # Dream-7B exposes mask_token_id in model config; tokenizer may not set it.
    mask_id = next(
        (v for v in [tokenizer.mask_token_id,
                     getattr(drafter.config, "mask_token_id", None),
                     tokenizer.unk_token_id]
         if v is not None),
        None,
    )
    if mask_id is None:
        raise ValueError(
            "Cannot determine mask token ID. "
            "Neither tokenizer.mask_token_id nor drafter.config.mask_token_id is set."
        )
    vocab_size = tokenizer.vocab_size

    stop_ids: set = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    try:
        gen_eos = target.generation_config.eos_token_id
        stop_ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    except Exception:
        pass
    stop_ids.discard(None)

    context   = prompt_ids.to(device)
    generated = []
    total_draft_accepted = 0
    n_steps   = 0
    t0 = time.perf_counter()

    while len(generated) < max_new_tokens:
        k = min(draft_len, max_new_tokens - len(generated))

        draft_ids                        = dlm_draft(drafter, context, k, mask_id, vocab_size)
        accepted, n_acc, hit_eos         = ar_verify(target, context, draft_ids, stop_ids)

        tokens = accepted.tolist()
        generated.extend(tokens)
        context = torch.cat([context, accepted])
        total_draft_accepted += n_acc
        n_steps += 1

        if hit_eos or stop_ids.intersection(tokens):
            break

    elapsed = time.perf_counter() - t0
    mat = total_draft_accepted / n_steps if n_steps > 0 else 0.0
    print(f"\n[stats] {len(generated)} tokens | {elapsed:.2f}s | "
          f"{len(generated)/elapsed:.1f} tok/s | steps={n_steps} | MAT={mat:.2f}")

    return torch.tensor(generated, dtype=torch.long)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Naive DLM speculative decoding")
    p.add_argument("--prompt",         required=True)
    p.add_argument("--target",         default="Qwen/Qwen3-8B")
    p.add_argument("--drafter",        default="dream-org/dream-v0-instruct-7b")
    p.add_argument("--draft-len",      type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device",         default="cuda")
    args = p.parse_args()

    tokenizer, target, drafter = load_models(
        args.target, args.drafter, args.device
    )

    messages     = [{"role": "user", "content": args.prompt}]
    prompt_text  = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids   = tokenizer(prompt_text, return_tensors="pt").input_ids[0]

    print(f"\nPrompt: {args.prompt}\n")
    generated_ids = generate(
        target, drafter, tokenizer, prompt_ids,
        draft_len=args.draft_len,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
