# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Builder for SmolVLA (lerobot) under π_RL.

Loads `HuggingFaceVLA/smolvla_libero` (or any lerobot SmolVLAPolicy checkpoint)
via lerobot's own from_pretrained, then wraps it in
SmolVLAForRLActionPrediction. This avoids reimplementing the state_dict layout
and normalization that lerobot ships with the checkpoint.
"""

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build a SmolVLA PIRL adapter from a Hydra cfg block.

    Reads PIRL knobs from `cfg.smolvla.*` (mirrors openpi's `cfg.openpi.*`),
    with top-level `model_path`, `num_steps`, `num_action_chunks`, `action_dim`,
    `add_value_head`. Forwards every adapter-known key into rl_cfg so the
    SmolVLAForRLActionPrediction.__init__ flag surface is fully exposed.
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from rlinf.models.embodiment.smolvla.smolvla_action_model import (
        SmolVLAForRLActionPrediction,
    )

    if not getattr(cfg, "model_path", None):
        raise ValueError("smolvla builder requires cfg.model_path (HF repo or local dir).")

    ckpt_path = str(cfg.model_path)
    lerobot_policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    if torch_dtype is not None:
        # Cast the whole policy (VLM + action expert + action_in_proj +
        # action_out_proj) so FSDP sees uniform dtype across all params it
        # wraps. lerobot ships the VLM in bf16 by default but action_in_proj
        # / action_out_proj are fp32 — without this cast FSDP wrap_model
        # crashes with "Must flatten tensors with uniform dtype". Choose
        # precision in YAML (precision: "bf16" for memory, "float32" for
        # accuracy).
        lerobot_policy = lerobot_policy.to(dtype=torch_dtype)

    # Sub-block carries SmolVLA-specific PIRL knobs (mirrors cfg.openpi). Use
    # getattr with default {} so missing sub-blocks don't crash.
    sub = getattr(cfg, "smolvla", {}) or {}

    def _g(key, default):
        # Prefer the smolvla.* sub-block, fall back to top-level for keys that
        # YAML conventionally puts at top level (num_steps, add_value_head, ...).
        if hasattr(sub, "get") and key in sub:
            return sub.get(key)
        return cfg.get(key, default)

    rl_cfg = {
        # noise schedule — default flow_sde (the adapter rejects flow_ode at train time).
        "noise_method": _g("noise_method", "flow_sde"),
        "noise_level": _g("noise_level", 0.5),
        "num_steps": _g("num_steps", lerobot_policy.config.num_steps),
        # value / critic
        "add_value_head": _g("add_value_head", True),
        "value_after_vlm": _g("value_after_vlm", False),
        "chunk_critic_input": _g("chunk_critic_input", True),
        "detach_critic_input": _g("detach_critic_input", False),
        # logp / entropy / loss reduction
        "safe_get_logprob": _g("safe_get_logprob", False),
        "joint_logprob": _g("joint_logprob", False),
        "ignore_last": _g("ignore_last", False),
        # training scope
        "train_expert_only": _g("train_expert_only", True),
        # action dims (env dim used to slice padding before PPO ratio)
        "num_action_chunks": _g("num_action_chunks", lerobot_policy.config.chunk_size),
        "action_env_dim": _g("action_env_dim",
            cfg.get("action_dim", lerobot_policy.config.action_feature.shape[0])),
    }

    # Pass the ckpt path so the wrapper can load policy_preprocessor /
    # policy_postprocessor (lerobot >= Q3-2025 format).
    model = SmolVLAForRLActionPrediction(lerobot_policy, rl_cfg=rl_cfg, ckpt_path=ckpt_path)

    # Mirrors openpi:66-67 — when train_expert_only, freeze VLM so only the
    # action expert + value/noise heads receive gradients.
    if rl_cfg["train_expert_only"]:
        model.freeze_vlm()

    model.eval()
    return model
