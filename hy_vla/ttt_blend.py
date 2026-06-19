"""Velocity-blend (manual CFG) sampling toward the language prior.

Shared by the offline diagnostic (``scripts/step1_velocity_blend.py``) and the
RoboTwin eval wrapper (``robotwin_eval/policy_wrapper.py``), so both paths use
one validated implementation.

Per Euler step the action expert is denoised twice — against the FULL-vision
prefix (``v_full``) and against the all-masked prefix (``v_masked``) — and the
velocities are linearly blended::

    v = w * v_full + (1 - w) * v_masked

    w = 1 -> stock sampling (vision fully trusted); fp-exact vs sample_actions
    w = 0 -> pure language prior
    w < 1 -> drift toward the language prior (less vision)

The masked branch sets every per-camera ``img_mask`` to False, which drops the
image patch tokens from attention (``embed_prefix``); step 0 validated that this
yields a real, instruction-dependent language prior on Hy-VLA despite the model
having no condition-dropout training.
"""

from __future__ import annotations

import torch

from hy_vla.modeling_hy_vla import make_att_2d_masks


@torch.no_grad()
def build_prefix(model, images, img_masks, lang_tokens, lang_masks):
    """Replicate the prefix/KV-cache stage of ``sample_actions``; returns
    ``(prefix_pad_masks, past_key_values)`` ready for ``denoise_step``."""
    pe, ppm, pam, mmp, iir, ifr = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    att2d = make_att_2d_masks(ppm, pam)
    pos = torch.cumsum(ppm, dim=1) - 1
    model._apply_visual_segment_mask(att2d, iir, ifr)
    (_, _), pkv, _, _ = model.dual_tower.forward(
        attention_mask=att2d,
        position_ids=pos,
        past_key_values=None,
        inputs_embeds=[pe, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
        modality_masks=[mmp, None],
    )
    return ppm, pkv


@torch.no_grad()
def sample_actions_blend(model, images, img_masks, lang_tokens, lang_masks, state,
                         w, noise=None, return_traj=False):
    """Euler-integrate the blended velocity field.

    ``model`` is the inner ``HyVLAFlowMatching`` (i.e. ``policy.model``).
    Returns the action tensor ``(B, n_action_steps, max_action_dim)``; if
    ``return_traj`` also returns a per-step list of
    ``(‖v_full‖, ‖v_masked‖, ‖v_blend‖)`` mean-per-motor L2 norms.
    """
    bsize = state.shape[0]
    device = state.device
    if noise is None:
        noise = model.sample_noise(
            (bsize, model.config.n_action_steps, model.config.max_action_dim), device
        )

    ppm_f, pkv_f = build_prefix(model, images, img_masks, lang_tokens, lang_masks)
    masked = [torch.zeros_like(m) for m in img_masks]
    ppm_m, pkv_m = build_prefix(model, images, masked, lang_tokens, lang_masks)

    dt = torch.tensor(-1.0 / model.config.num_steps, dtype=torch.float32, device=device)
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    def _n(v):
        return v.detach().float().norm(dim=-1).mean().item()

    traj = []
    while time >= -dt / 2:
        et = time.expand(bsize)
        v_f, _ = model.denoise_step(state, ppm_f, pkv_f, x_t, et)
        v_m, _ = model.denoise_step(state, ppm_m, pkv_m, x_t, et)
        # lerp form: endpoints w=1 -> v_f and w=0 -> v_m are fp-exact
        v = w * v_f + (1.0 - w) * v_m
        if return_traj:
            traj.append((_n(v_f), _n(v_m), _n(v)))
        x_t = x_t + dt * v
        time = time + dt

    return (x_t, traj) if return_traj else x_t


__all__ = ["build_prefix", "sample_actions_blend"]
