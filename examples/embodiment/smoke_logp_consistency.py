"""Smoke test: sample-time vs rescore-time logp consistency in bf16.

Why this exists
---------------
Phase 4 GRPO training reports `actor/approx_kl ~ -0.25` from step 1, with
±0.05 step-to-step variance. We need to know whether that bias reflects:

  (i)  a real per-element bug in the adapter (sample_actions stores a logp
       that the actor's get_log_prob_value path cannot reproduce), OR
  (ii) standard GRPO instability under sparse reward (which kl_beta can fix).

The training metric `approx_kl` aggregates over masked positions in a way
that's hard to reverse-engineer without running the framework. This script
bypasses RLinf entirely and tests the adapter math directly: same chains,
same denoise_inds, same observations, two paths (sample-time vs rescore),
two modes (eval/eval vs eval/train).

What it tests
-------------
- Test A (eval, eval): sample_actions in eval mode, then default_forward
  in eval mode on the same chains. Diff should be near zero — same model,
  same inputs, deterministic flow_sde with stored chains.

- Test B (eval, train): sample_actions in eval mode, then default_forward
  in train mode. Diff should match Test A if SmolVLA has no mode-dependent
  layers (no dropout, no BatchNorm — empirically true). If Test B diverges
  but Test A is clean, the bug is mode-dependent (need to enforce eval at
  rescore time).

Interpretation
--------------
| diff_A   | diff_B   | meaning                                                |
|----------|----------|--------------------------------------------------------|
| ~0       | ~0       | adapter math is consistent in bf16; approx_kl=-0.25 in |
|          |          | training is downstream (FSDP, AMP, or GRPO instability)|
| ~0       | large    | train mode introduces a divergence (dropout etc)       |
| large    | large    | per-element adapter bug; sample_actions and            |
|          |          | get_log_prob_value disagree on the same chains         |

Run
---
From RLinf root, with venv active and HF_ENDPOINT set:

    cd /root/autodl-tmp/workspace/RLinf
    python3 examples/embodiment/smoke_logp_consistency.py

Takes ~30s for model download + forward; ~1 GB GPU memory in bf16.
"""

import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.smolvla import get_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _build_fake_batch():
    # Mirror what env_worker + obs_processor produce after PolicyProcessor:
    # main + wrist camera (256×256 RGB normalized), 8-dim Franka state, and
    # a tokenized task instruction. Lengths chosen to look like a real
    # LIBERO task ("pick up the milk and put it on the plate" tokenizes to
    # ~17 BPE tokens in SmolVLM2-500M-Video-Instruct).
    torch.manual_seed(42)
    return {
        "observation.images.image": torch.randn(1, 3, 256, 256, dtype=DTYPE, device=DEVICE),
        "observation.images.image2": torch.randn(1, 3, 256, 256, dtype=DTYPE, device=DEVICE),
        "observation.state": torch.randn(1, 8, dtype=DTYPE, device=DEVICE),
        "observation.language.tokens": torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],
            dtype=torch.long,
            device=DEVICE,
        ),
        "observation.language.attention_mask": torch.ones(1, 17, dtype=torch.bool, device=DEVICE),
        "task": ["pick up the milk and put it on the plate"],
    }


def _build_cfg():
    return OmegaConf.create({
        "model_path": "HuggingFaceVLA/smolvla_libero",
        "precision": "bf16",
        "num_action_chunks": 50,
        "action_dim": 7,
        "use_proprio": True,
        "num_steps": 4,
        "add_value_head": False,
        "smolvla": {
            "num_images_in_input": 2,
            "noise_level": 0.5,
            "action_chunk": 50,
            "num_steps": 4,
            "train_expert_only": True,
            "action_env_dim": 7,
            "noise_method": "flow_sde",
            "add_value_head": False,
            "value_after_vlm": False,
            "chunk_critic_input": True,
            "detach_critic_input": True,
            "safe_get_logprob": False,
            "joint_logprob": False,
            "ignore_last": False,
        },
    })


def _run():
    print(f"[setup] device={DEVICE}, dtype={DTYPE}")
    cfg = _build_cfg()
    print(f"[setup] loading SmolVLA from {cfg.model_path} ...")
    policy = get_model(cfg, torch_dtype=DTYPE).to(DEVICE)
    batch = _build_fake_batch()
    print(f"[setup] fake batch built, language tokens shape={batch['observation.language.tokens'].shape}")

    # ----- Sample-time path (always in eval; this is what rollout does) -----
    policy.eval()
    with torch.no_grad():
        out_sample = policy.sample_actions(batch, mode="train", compute_values=False)
    chains = out_sample["chains"]
    denoise_inds = out_sample["denoise_inds"]
    prev_logp = out_sample["prev_logprobs"]
    print(f"[sample] chains.shape={tuple(chains.shape)}, prev_logp.shape={tuple(prev_logp.shape)}, dtype={prev_logp.dtype}")

    # Build forward_inputs blob the actor worker passes back to default_forward.
    forward_inputs = {"chains": chains, "denoise_inds": denoise_inds}
    for k, v in batch.items():
        if torch.is_tensor(v):
            forward_inputs[k] = v

    # ----- Test A: rescore in eval mode -----
    policy.eval()
    with torch.no_grad():
        out_A = policy(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
            use_cache=False,
        )
    rescore_A = out_A["logprobs"]
    print(f"[A]      rescore.shape={tuple(rescore_A.shape)}, dtype={rescore_A.dtype}")

    if prev_logp.shape != rescore_A.shape:
        print(f"[A] WARN shape mismatch: prev={tuple(prev_logp.shape)} vs rescore={tuple(rescore_A.shape)} — comparison may be misleading")

    diff_A = (prev_logp.float() - rescore_A.float()).abs()
    print(
        f"[A] eval/eval  diff:  max={diff_A.max().item():.6f}  "
        f"mean={diff_A.mean().item():.6f}  std={diff_A.std().item():.6f}"
    )

    # ----- Test B: rescore in train mode (matches what fsdp_actor_worker does) -----
    policy.train()
    with torch.no_grad():
        out_B = policy(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
            use_cache=False,
        )
    rescore_B = out_B["logprobs"]
    diff_B = (prev_logp.float() - rescore_B.float()).abs()
    print(
        f"[B] eval/train diff:  max={diff_B.max().item():.6f}  "
        f"mean={diff_B.mean().item():.6f}  std={diff_B.std().item():.6f}"
    )

    # ----- Sanity check C: perturb chains_next → rescore SHOULD diverge -----
    # If [A]/[B] are exactly 0 we want to rule out a no-op comparison. Add
    # small noise to the stored chains and verify the rescore path picks it
    # up — if diff_C also reports ~0, the test isn't actually testing
    # anything (e.g. cached logp, broken fwd, or shared tensor reference).
    policy.eval()
    chains_perturbed = chains.clone()
    chains_perturbed[:, -1] = chains_perturbed[:, -1] + 0.05  # final chain state shift
    fwd_perturbed = dict(forward_inputs)
    fwd_perturbed["chains"] = chains_perturbed
    with torch.no_grad():
        out_C = policy(
            forward_inputs=fwd_perturbed,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
            use_cache=False,
        )
    rescore_C = out_C["logprobs"]
    diff_C = (prev_logp.float() - rescore_C.float()).abs()
    print(
        f"[C] perturbed diff:    max={diff_C.max().item():.6f}  "
        f"mean={diff_C.mean().item():.6f}  std={diff_C.std().item():.6f}"
    )

    # ----- Sanity check D: different batch → completely different logp -----
    # Swap language tokens for a totally different sentence. If diff_D is
    # ~0 the rescore isn't even reading the batch (would mean global caching
    # or the prefix cache is detached from the actual obs).
    other_batch = {**batch}
    other_batch["observation.language.tokens"] = torch.tensor(
        [[50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]],
        dtype=torch.long,
        device=DEVICE,
    )
    fwd_other = {"chains": chains, "denoise_inds": denoise_inds}
    for k, v in other_batch.items():
        if torch.is_tensor(v):
            fwd_other[k] = v
    with torch.no_grad():
        out_D = policy(
            forward_inputs=fwd_other,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
            use_cache=False,
        )
    rescore_D = out_D["logprobs"]
    diff_D = (prev_logp.float() - rescore_D.float()).abs()
    print(
        f"[D] other-task diff:   max={diff_D.max().item():.6f}  "
        f"mean={diff_D.mean().item():.6f}  std={diff_D.std().item():.6f}"
    )

    # ----- Conclusion -----
    THRESH = 0.01
    SANITY_MIN = 1e-4  # Test C/D MUST show real divergence
    sanity_ok = diff_C.max().item() > SANITY_MIN and diff_D.max().item() > SANITY_MIN
    if not sanity_ok:
        print("\n=== SANITY CHECK FAILED ===")
        print(f"  diff_C max = {diff_C.max().item():.6f} (expected > {SANITY_MIN})")
        print(f"  diff_D max = {diff_D.max().item():.6f} (expected > {SANITY_MIN})")
        print("  → Either rescore is a no-op (cached logp, wrong fwd entry point)")
        print("     OR shared tensor reference makes A/B/C/D return the same buffer.")
        print("  → Cannot trust A/B results until this is debugged.")
        return 1
    print(f"\n[sanity] C and D both show real divergence (max diff > {SANITY_MIN}) — test IS doing work.")

    print("\n=== Conclusion ===")
    if diff_A.mean().item() < THRESH and diff_B.mean().item() < THRESH:
        print("PASS: sample-time and rescore-time logp agree at the element level in bf16.")
        print("  → approx_kl=-0.25 in Phase 4 training is NOT a per-element adapter bug.")
        print("  → The bias is either (1) downstream (FSDP all_gather / amp_context numerical")
        print("     noise across rollout-side vs actor-side forwards), or (2) just standard")
        print("     GRPO instability under sparse binary reward.")
        print("  → kl_beta=0.01 + larger group are reasonable interventions; Phase 5 results")
        print("     can be trusted.")
    elif diff_A.mean().item() < THRESH and diff_B.mean().item() >= THRESH:
        print("PARTIAL: eval/eval matches but eval/train diverges.")
        print("  → A mode-dependent op (dropout / BN / Dropout-like) is active in train mode.")
        print("  → Fix: enforce model.eval() inside get_log_prob_value, or audit SmolVLM2")
        print("     submodules for train-time stochasticity.")
    else:
        print("FAIL: per-element bug in adapter sample/rescore paths.")
        print(f"  → max diff = {max(diff_A.max().item(), diff_B.max().item()):.6f}")
        print("  → Phase 5 results would be unreliable until this is fixed.")
        print("  → Audit adapter sample_actions vs get_log_prob_value: chains storage,")
        print("     denoise_inds selection, sample_mean_var_val determinism.")

    return 0


if __name__ == "__main__":
    sys.exit(_run())
