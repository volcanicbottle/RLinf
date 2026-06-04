# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""π_RL action-model wrapper around lerobot's SmolVLA.

Day-3 scope (this file currently implements only):
    - prepare_observations: RLinf env_obs dict -> lerobot batch dict.
    - sample_actions_with_chains: deterministic Euler with chain recording (Edit 5a).
    - predict_action_batch: rollout entrypoint; flow_ode only.

W1 scope (NotImplementedError stubs below):
    - sample_actions_stochastic (flow_sde / flow_noise): Edit 5b.
    - get_log_prob: scoring path for PPO.
    - value_head / noise_head: critic + learned σ.

Conventions (pinned — match openpi_action_model.py and lerobot modeling_smolvla.py):
    SmolVLA flow direction: x_1 = noise, x_0 = data action.
    x_t = t * x_1 + (1-t) * x_0; v_t = x_1 - x_0 (constant velocity flow matching).
    Integration runs t: 1.0 -> 1/N (DECREASING). dt = -1/N (negative).
    Timesteps grid: torch.linspace(1, 1/N, N) ++ [0.0]  -> length N+1.
    delta = timesteps[idx] - timesteps[idx+1]   (positive).
    Per-step update (ODE): x_{idx+1} = x_t + dt * v_t  (matches lerobot modeling_smolvla.py:876).
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class SmolVLAForRLActionPrediction(nn.Module, BasePolicy):
    """RL wrapper around lerobot's SmolVLAPolicy.

    Owns the inner lerobot policy and exposes the π_RL interface
    (predict_action_batch, default_forward, sample_actions_with_chains).
    """

    def __init__(
        self,
        lerobot_policy,
        rl_cfg: dict | None = None,
        ckpt_path: str | None = None,
    ):
        nn.Module.__init__(self)
        self.policy = lerobot_policy           # the lerobot SmolVLAPolicy
        self.inner = lerobot_policy.model      # the VLAFlowMatching
        self.rl_cfg = rl_cfg or {}

        # PIRL config flags (mirrors openpi_action_model / dexbotic_pi config surface).
        cfg = self.rl_cfg
        self.num_steps: int = int(cfg.get("num_steps", self.inner.config.num_steps))
        self.action_env_dim: int = int(cfg.get("action_env_dim",
            self.policy.config.action_feature.shape[0]))
        self.num_action_chunks: int = int(cfg.get("num_action_chunks",
            self.inner.config.chunk_size))
        self.noise_method: str = str(cfg.get("noise_method", "flow_sde"))  # flow_sde|flow_cps|flow_noise
        self.noise_level: float = float(cfg.get("noise_level", 0.5))
        self.joint_logprob: bool = bool(cfg.get("joint_logprob", False))
        self.ignore_last: bool = bool(cfg.get("ignore_last", False))
        self.add_value_head: bool = bool(cfg.get("add_value_head", True))
        self.value_after_vlm: bool = bool(cfg.get("value_after_vlm", False))  # SmolVLA: only suffix path supported
        self.chunk_critic_input: bool = bool(cfg.get("chunk_critic_input", True))
        self.detach_critic_input: bool = bool(cfg.get("detach_critic_input", False))
        self.safe_get_logprob: bool = bool(cfg.get("safe_get_logprob", False))
        self.train_expert_only: bool = bool(cfg.get("train_expert_only", True))

        # Suffix hidden size = expert hidden size. action_out_proj output dim = max_action_dim.
        expert_hidden = self.inner.vlm_with_expert.expert_hidden_size
        max_action_dim = self.inner.config.max_action_dim

        # Value head: maps mean-pooled suffix features -> scalar value. Only built if
        # add_value_head is set. Matches dexbotic_pi `value_head` placement.
        if self.add_value_head:
            self.value_head = nn.Linear(expert_hidden, 1)
        else:
            self.value_head = None

        # Noise head (for noise_method="flow_noise"): predicts per-token σ from suffix.
        if self.noise_method == "flow_noise":
            self.noise_head = nn.Linear(expert_hidden, max_action_dim)
        else:
            self.noise_head = None

        # Image / state keys the loaded SFT ckpt expects, derived from the
        # lerobot config. Used by prepare_observations.
        self._image_feature_keys = list(self.policy.config.image_features.keys())
        self._state_feature_key = "observation.state"
        # Tokenizer for language. Pulled from the inner SmolVLM expert.
        self._tokenizer = self.inner.vlm_with_expert.processor.tokenizer

        # Load lerobot's policy_preprocessor / policy_postprocessor pipelines if
        # the SFT checkpoint shipped them (smolvla_libero ships both since the
        # 2025-Q3 lerobot processor refactor). These contain:
        #   pre:  RenameObservations + AddBatchDim + NewLineTask + Tokenizer
        #         + Device + Normalizer.
        #   post: Unnormalizer + DeviceProcessor.
        # When present they are the source of truth for action/state stats —
        # exactly the stats the SFT was trained with.
        self.preprocessor = None
        self.postprocessor = None
        if ckpt_path is not None:
            from lerobot.processor import (
                PolicyProcessorPipeline,
                policy_action_to_transition,
                transition_to_policy_action,
            )
            from lerobot.utils.constants import (
                POLICY_POSTPROCESSOR_DEFAULT_NAME,
                POLICY_PREPROCESSOR_DEFAULT_NAME,
            )
            try:
                # Preprocessor: dict -> dict (default converters).
                self.preprocessor = PolicyProcessorPipeline.from_pretrained(
                    ckpt_path,
                    config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                )
                # Postprocessor: PolicyAction (Tensor) -> PolicyAction. Must
                # override the default batch converters to action-only ones
                # (matches make_smolvla_pre_post_processors).
                self.postprocessor = PolicyProcessorPipeline.from_pretrained(
                    ckpt_path,
                    config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
                    to_transition=policy_action_to_transition,
                    to_output=transition_to_policy_action,
                )
            except (FileNotFoundError, ValueError):
                # ckpt does not ship processors (older format). Caller can supply
                # stats explicitly in W1 or live with un-normalized I/O for now.
                self.preprocessor = None
                self.postprocessor = None

    # ---------------------------------------------------------------- adapters
    def rename_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Step 1 of the adapter: rename RLinf env_obs keys to lerobot canonical keys.

        RLinf LIBERO env_obs (rlinf/envs/libero/libero_env.py:580):
            {"main_images": [B,3,H,W] float32 (or uint8),
             "wrist_images": [B,3,H,W],
             "states": [B,8] float32,                            # lerobot 8-D LIBERO convention
             "task_descriptions": list[str]}

        Returns a dict keyed by `policy.config.image_features.keys()` +
        "observation.state" + "task". Image dtype is left untouched; the
        downstream preprocessor (or our fallback path) handles range scaling.
        """
        img_keys = self._image_feature_keys
        assert len(img_keys) >= 1, "SmolVLA config has no image features."

        out: dict[str, Any] = {}
        out[img_keys[0]] = env_obs["main_images"]
        if "wrist_images" in env_obs and len(img_keys) > 1:
            out[img_keys[1]] = env_obs["wrist_images"]

        out[self._state_feature_key] = env_obs["states"]
        out["task"] = env_obs.get("task_descriptions", [""] * env_obs["states"].shape[0])
        return out

    def prepare_observations(self, env_obs: dict[str, Any]) -> dict[str, Tensor]:
        """Full adapter: rename keys, then run lerobot's preprocessor pipeline if
        loaded (tokenize + normalize + device move). If the SFT ckpt did not ship
        a preprocessor, fall back to manual tokenization with NO normalization.

        Returns a batch dict keyed for `VLAFlowMatching.embed_prefix` /
        `prepare_images` / `prepare_state` consumption.
        """
        device = next(self.parameters()).device
        renamed = self.rename_env_obs(env_obs)

        if self.preprocessor is not None:
            # lerobot's preprocessor runs F.interpolate / Resize on images which
            # don't support uint8 inputs (RuntimeError: "upsample_bilinear2d_out_frame"
            # not implemented for 'Byte'). LIBERO env returns uint8 RGB in HWC
            # numpy convention [B, H, W, 3]; lerobot/SmolVLM expects CHW [B, 3,
            # H, W]. Cast uint8→float AND permute HWC→CHW before handing off,
            # otherwise the preprocessor's Resize misinterprets dims and Conv2d
            # crashes with "expected input to have 3 channels, but got 256".
            for k in list(renamed.keys()):
                v = renamed[k]
                if torch.is_tensor(v) and v.dtype == torch.uint8:
                    v = v.float() / 255.0
                if torch.is_tensor(v) and v.dim() == 4 and v.shape[-1] == 3 and v.shape[1] != 3:
                    v = v.permute(0, 3, 1, 2).contiguous()
                renamed[k] = v
            batch = self.preprocessor(renamed)
            # The preprocessor moves to device + normalizes + tokenizes. Output
            # keys: observation.images.<k> (in [-1,1] range only if Normalize is
            # configured to do so; otherwise raw — SmolVLA does its own [-1,1]
            # rescale inside prepare_images), observation.state (normalized),
            # observation.language.* (tokenized), task (string), action (absent here).
            # Cast all floating-point inputs to match model dtype. With
            # precision=bf16 the model weights are bf16 but env returns state
            # in fp32 — without this cast state_proj's F.linear crashes with
            # "mat1 and mat2 must have the same dtype".
            model_dtype = next(self.parameters()).dtype
            for k, v in batch.items():
                if torch.is_tensor(v) and v.is_floating_point() and v.dtype != model_dtype:
                    batch[k] = v.to(model_dtype)
            return batch

        # ---- fallback: manual tokenization, no normalization. Same as before.
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch: dict[str, Any] = {}
        for k, v in renamed.items():
            if k == "task":
                continue
            tensor = v if torch.is_tensor(v) else torch.as_tensor(v)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            batch[k] = tensor.to(device)

        task_strs = list(renamed["task"])
        tokenized = self._tokenizer(
            task_strs,
            padding="longest",
            truncation=True,
            max_length=self.policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].to(device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(device).bool()
        return batch

    # --------------------------------------------------------- core rollout API
    @torch.no_grad()
    def sample_actions_with_chains(
        self,
        batch: dict[str, Tensor],
        num_steps: int | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Deterministic Euler rollout that records intermediate states (Edit 5a).

        Mirrors lerobot VLAFlowMatching.sample_actions (modeling_smolvla.py:812)
        line-for-line, adds chain capture. NO σ, NO Gaussian noise. Day-3 path.

        Returns
        -------
        actions : [B, chunk_size, max_action_dim]   final x_t
        chains  : [B, N+1, chunk_size, max_action_dim]  recorded states (chains[:, 0] = noise)
        timesteps : [N+1]   grid [1.0, 1-1/N, ..., 1/N, 0.0]  (last entry is the post-final t)
        """
        model = self.inner
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = state.shape[0]
        device = state.device

        N = num_steps if num_steps is not None else model.config.num_steps
        if noise is None:
            actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
            noise = model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / N
        # Grid matches openpi (linspace then append 0). For N=10:
        # [1.0, 0.9, ..., 0.1, 0.0]  length N+1.
        timesteps = torch.cat(
            [torch.linspace(1.0, 1.0 / N, N, device=device),
             torch.tensor([0.0], device=device)],
            dim=0,
        )

        x_t = noise
        chains = [x_t.clone()]
        for step in range(N):
            t_scalar = 1.0 + step * dt                  # MATCH lerobot:849 exactly
            time_tensor = torch.tensor(t_scalar, dtype=torch.float32, device=device).expand(bsize)
            v_t = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
            chains.append(x_t.clone())

        chains = torch.stack(chains, dim=1)              # [B, N+1, H, A]
        return x_t, chains, timesteps

    # ============================== PIRL training surface ==============================
    # Methods below mirror dexbotic_pi_policy.py PIRL contract, ported to SmolVLA's
    # VLAFlowMatching API (state lives in prefix, embed_suffix takes only noisy_actions+t).

    def _build_prefix_cache(self, batch: dict[str, Tensor]):
        """Build prefix embeddings + KV cache from a prepared batch.

        Returns (prefix_pad_masks, past_key_values) — the two outputs every
        denoise step needs. Mirrors the prefix block of sample_actions_with_chains.
        """
        model = self.inner
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def get_suffix_out(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """Run the action expert for one denoising step and return (v_t, suffix_out).

        Thin wrapper over VLAFlowMatching.denoise_step(return_suffix_out=True) —
        the lerobot Edit 4 hook. Matches dexbotic_pi.get_suffix_out's role but
        returns v_t too (since SmolVLA computes it inline).
        """
        bsize = x_t.shape[0]
        device = x_t.device
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        if timestep.dim() == 0:
            timestep = timestep.expand(bsize)
        v_t, suffix_out = self.inner.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
            return_suffix_out=True,
        )
        return v_t, suffix_out

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        prefix_pad_masks,
        past_key_values,
        mode: str,
        denoise_steps: int,
        compute_values: bool = True,
    ):
        """One flow-matching step: return (x_t_mean, x_t_std, value_t, v_t).

        Ported from dexbotic_pi.sample_mean_var_val. Removes the `state` arg
        because SmolVLA puts state in the prefix cache (already done in
        _build_prefix_cache). Adds v_t to the return tuple so sample_actions
        can record chains without a second forward.
        """
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)
        noise_level = torch.tensor(self.noise_level, device=device)

        # Force x_t to match action_in_proj's weight dtype before it goes
        # into lerobot's denoise_step → embed_suffix → action_in_proj. The
        # outer sample_actions sets x_dtype from action_in_proj at creation
        # time, but the rollout worker's path occasionally produces fp32 x_t
        # even when the policy is cast to bf16; this final cast is the
        # belt-and-suspenders fix for the F.linear mat1/mat2 dtype check.
        w_dtype = self.inner.action_in_proj.weight.dtype
        if x_t.dtype != w_dtype:
            x_t = x_t.to(dtype=w_dtype)

        # Grid: [1, (N-1)/N, ..., 1/N, 0] — length N+1.
        timesteps = torch.linspace(1.0, 1.0 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        v_t, suffix_out = self.get_suffix_out(
            prefix_pad_masks, past_key_values, x_t, t_input,
        )

        # Suffix-based value (matches dexbotic_pi when value_after_vlm=False).
        if self.add_value_head and compute_values and not self.value_after_vlm:
            if self.chunk_critic_input:
                suffix_for_value = torch.mean(
                    suffix_out[:, : self.inner.config.chunk_size], dim=1, keepdim=False
                )
            else:
                suffix_for_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.detach_critic_input:
                suffix_for_value = suffix_for_value.detach()
            value_t = self.value_head(suffix_for_value.to(self.value_head.weight.dtype))[:, 0]
        else:
            value_t = torch.zeros((bsize,), device=device)

        # x0/x1 prediction from constant-velocity flow matching.
        delta_b = delta[:, None, None].expand_as(x_t)
        t_b = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_b
        x1_pred = x_t + v_t * (1.0 - t_b)

        if mode == "eval":
            x0_weight = 1.0 - (t_b - delta_b)
            x1_weight = t_b - delta_b
            x_t_std = torch.zeros_like(t_b)
        elif mode == "train":
            if self.noise_method == "flow_sde":
                denom_t = torch.where(timesteps == 1, timesteps[1], timesteps)
                sigmas = noise_level * torch.sqrt(timesteps / (1.0 - denom_t))[:-1]
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_b) - (t_b - delta_b)
                x1_weight = t_b - delta_b - sigma_i**2 * delta_b / (2.0 * t_b)
                x_t_std = torch.sqrt(delta_b) * sigma_i
            elif self.noise_method == "flow_cps":
                cos_term = torch.cos(math.pi * noise_level / 2.0).to(device)
                sin_term = torch.sin(math.pi * noise_level / 2.0).to(device)
                x0_weight = torch.ones_like(t_b) - (t_b - delta_b)
                x1_weight = (t_b - delta_b) * cos_term
                x_t_std = (t_b - delta_b) * sin_term
            elif self.noise_method == "flow_noise":
                x0_weight = 1.0 - (t_b - delta_b)
                x1_weight = t_b - delta_b
                x_t_std = self.noise_head(suffix_out.to(self.noise_head.weight.dtype))
            else:
                raise ValueError(f"Invalid noise_method: {self.noise_method}")
        else:
            raise ValueError(f"Invalid mode: {mode!r} (expected 'train'|'eval')")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t, v_t

    def get_logprob_norm(self, sample, mu, sigma):
        """Per-element Gaussian log-density, with sigma==0 short-circuited to 0."""
        if self.safe_get_logprob:
            return -torch.pow(sample - mu, 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        return 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))

    def get_log_prob_value(
        self,
        batch: dict[str, Tensor],
        chains: Tensor,
        denoise_inds: Tensor,
        compute_values: bool = False,
    ):
        """Compute per-step (log_prob, value, entropy) for stored chains.

        Mirrors dexbotic_pi.get_log_prob_value. Builds prefix once (no_grad when
        train_expert_only=True), then loops over `denoise_inds` to evaluate
        sample_mean_var_val + get_logprob_norm at each (step, chain_idx) pair.
        """
        bsize = chains.shape[0]
        no_grad_ctx = torch.no_grad() if self.train_expert_only else torch.enable_grad()
        with no_grad_ctx:
            prefix_pad_masks, past_key_values = self._build_prefix_cache(batch)

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.joint_logprob:
            num_steps = self.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0], torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t, _ = self.sample_mean_var_val(
                chains_pre, denoise_ind, prefix_pad_masks, past_key_values,
                "train", self.num_steps, compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            chains_values.append(value_t)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        if self.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        mode: str = "train",
        compute_values: bool = True,
        noise: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Full PIRL rollout: builds chains, prev_logprobs, prev_values, denoise_inds.

        Mirrors dexbotic_pi.sample_actions. mode="train" injects stochasticity at
        a random denoise step; mode="eval" runs pure ODE.
        Returned tensors are NOT sliced to action_env_dim (caller does that).
        """
        was_training = self.training
        self.eval()
        try:
            prefix_pad_masks, past_key_values = self._build_prefix_cache(batch)

            bsize = prefix_pad_masks.shape[0]
            device = prefix_pad_masks.device
            # action_in_proj is float32 (set at policy init, independent of VLM dtype);
            # match sample_noise's hard-coded float32 to avoid a dtype mismatch.
            x_dtype = self.inner.action_in_proj.weight.dtype
            N = self.num_steps
            chunk = self.inner.config.chunk_size
            max_a = self.inner.config.max_action_dim

            if noise is None:
                x_t = torch.randn(bsize, chunk, max_a, device=device, dtype=x_dtype)
            else:
                x_t = noise.to(device=device, dtype=x_dtype)

            chains = [x_t]
            log_probs = []
            values = []

            if self.joint_logprob:
                initial_log_prob = self.get_logprob_norm(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
                log_probs.append(initial_log_prob)

            # Build denoise_inds (which step gets the SDE/CPS noise).
            if mode == "train":
                if self.joint_logprob:
                    denoise_inds_row = torch.arange(N, device=device)
                else:
                    hi = N - 2 if self.ignore_last else N - 1
                    rand_idx = random.randint(0, hi)
                    denoise_inds_row = torch.tensor([rand_idx] * N, device=device)
            else:
                denoise_inds_row = torch.tensor([-1] * N, device=device)
            denoise_inds = denoise_inds_row[None].repeat(bsize, 1)

            for idx in range(N):
                step_mode = "train" if (mode == "train" and idx == int(denoise_inds[0, idx])) else "eval"
                x_t_mean, x_t_std, value_t, _ = self.sample_mean_var_val(
                    x_t, idx, prefix_pad_masks, past_key_values,
                    step_mode, N, compute_values,
                )
                x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
                log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
                values.append(value_t)
                chains.append(x_t)
                log_probs.append(log_prob)

            actions = x_t
            chains_t = torch.stack(chains, dim=1)  # [B, N+1, chunk, max_a]
            log_probs_t = torch.stack(log_probs, dim=1)[
                :, :, : self.num_action_chunks, : self.action_env_dim
            ]
            if self.joint_logprob:
                log_probs_t = log_probs_t.mean(dim=1)
            else:
                log_probs_t = log_probs_t[
                    torch.arange(log_probs_t.shape[0]), denoise_inds[:, 0]
                ]
            values_t = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

            return {
                "actions": actions,
                "chains": chains_t,
                "prev_logprobs": log_probs_t,
                "prev_values": values_t,
                "denoise_inds": denoise_inds,
            }
        finally:
            if was_training:
                self.train()

    # ============================== inference / rollout ===============================
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "train",
        compute_values: bool = True,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Any]]:
        """π_RL rollout entrypoint.

        Modes:
            mode="train"  → full PIRL rollout via sample_actions; result dict
                            contains prev_logprobs, prev_values, denoise_inds,
                            and a `forward_inputs` blob the actor worker re-feeds
                            into self.forward(forward_inputs=...) at training time.
            mode="eval"   → pure ODE; no logprobs, no values.

        Normalization: PolicyProcessorPipeline pre/post — see __init__ docstring.
        Output actions are in PHYSICAL scale (env consumes as-is).
        """
        batch = self.prepare_observations(env_obs)

        if mode == "eval":
            actions, chains, timesteps = self.sample_actions_with_chains(
                batch, num_steps=self.num_steps
            )
            actions_full = actions[:, :, : self.action_env_dim]
            actions_out = self.postprocessor(actions_full) if self.postprocessor is not None else actions_full
            return actions_out, {
                "chains": chains,
                "timesteps": timesteps,
                "prev_logprobs": None,
                "prev_values": None,
                "normalized": self.preprocessor is not None,
            }

        # mode == "train": full PIRL rollout with chains/logprobs/values.
        out = self.sample_actions(batch, mode="train", compute_values=compute_values)
        actions_full = out["actions"][:, :, : self.action_env_dim]
        actions_out = self.postprocessor(actions_full) if self.postprocessor is not None else actions_full

        # forward_inputs is what the actor worker passes back into self.forward()
        # during training to recompute logprobs for the proximal PPO ratio.
        forward_inputs = {
            "chains": out["chains"],
            "denoise_inds": out["denoise_inds"],
            "batch": batch,            # full prepared obs; default_forward unpacks it
        }
        result = {
            "chains": out["chains"],
            "prev_logprobs": out["prev_logprobs"],
            "prev_values": out["prev_values"],
            "denoise_inds": out["denoise_inds"],
            "forward_inputs": forward_inputs,
            "normalized": self.preprocessor is not None,
        }
        return actions_out, result

    # ============================== BasePolicy training dispatch ==============================
    def default_forward(self, data: dict, **kwargs) -> dict[str, Tensor]:
        """Training forward: recompute (logprobs, values, entropy) for stored chains.

        `data` is the `forward_inputs` blob produced by predict_action_batch:
            data["chains"]        — [B, N+1, chunk, max_a]
            data["denoise_inds"]  — [B, N]
            data["batch"]         — prepared obs dict (images, lang_tokens, state)
        Returns {"logprobs", "values", "entropy"} per actor-worker contract
        (rlinf/workers/actor/async_ppo_fsdp_worker.py).
        """
        compute_values = bool(kwargs.get("compute_values", False))
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        batch = data["batch"]

        log_probs, values, entropy = self.get_log_prob_value(
            batch, chains, denoise_inds, compute_values=compute_values,
        )

        # Slice padding → env dims; collapse step axis where appropriate.
        log_probs = log_probs[:, :, : self.num_action_chunks, : self.action_env_dim]
        entropy = entropy[:, :, : self.num_action_chunks, : self.action_env_dim]
        log_probs = log_probs.mean(dim=1)                             # [B, chunk, a_env]
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]  # [B, 1]
        values = values.mean(dim=-1, keepdim=False)                    # [B]
        return {"logprobs": log_probs, "values": values, "entropy": entropy}

    def forward(self, *args, forward_type=ForwardType.DEFAULT, **kwargs):
        """Dispatch on forward_type.

        Worker contract (obs 587): calls self.model(forward_inputs=...,
        compute_logprobs=True, compute_entropy=bool, compute_values=bool,
        use_cache=False). We resolve forward_inputs → data and route.
        """
        if "forward_inputs" in kwargs and "data" not in kwargs:
            kwargs["data"] = kwargs.pop("forward_inputs")

        if forward_type == ForwardType.DEFAULT or forward_type == "default_forward":
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"ForwardType={forward_type} not supported yet.")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        """Forward HF-style grad-ckpt enable to the inner VLM tower.

        RLinf's FSDPModelManager.setup_model_and_optimizer calls this on the
        top-level model. SmolVLA's wrapper is a plain nn.Module so the call
        misses; forward it to vlm_with_expert.vlm (which is a HF SmolVLM
        model that ships gradient_checkpointing_enable). The action expert
        is small enough that we don't checkpoint it.
        """
        vlm = self.inner.vlm_with_expert.vlm
        if hasattr(vlm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                vlm.gradient_checkpointing_enable()
            else:
                vlm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        vlm = self.inner.vlm_with_expert.vlm
        if hasattr(vlm, "gradient_checkpointing_disable"):
            vlm.gradient_checkpointing_disable()

    def freeze_vlm(self):
        """Freeze the VLM (vision encoder + LLM) leaving the action expert trainable.

        Mirrors dexbotic_pi.freeze_vlm. With train_expert_only=True, only the
        action expert layers + action_in_proj/action_out_proj + value_head/
        noise_head receive gradients.
        """
        # vlm_with_expert contains both the VLM and the action expert; the VLM
        # side is `.vlm`, the expert is `.lm_expert`. Freeze the VLM only.
        if hasattr(self.inner.vlm_with_expert, "vlm"):
            for p in self.inner.vlm_with_expert.vlm.parameters():
                p.requires_grad = False
        else:
            # Fallback: walk submodules and freeze anything not in the expert.
            for name, p in self.inner.vlm_with_expert.named_parameters():
                if "expert" not in name.lower():
                    p.requires_grad = False
