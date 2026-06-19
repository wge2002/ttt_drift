#!/usr/bin/env python
"""step 1 — manual CFG / velocity blend toward the language prior.

Step 0 (``scripts/diag_step0_mask.py``) showed the vision-masked branch is a
valid, instruction-dependent language prior on Hy-VLA. This script uses it to
build a guidance knob the base model does NOT ship (it has no CFG):

    every Euler step, run denoise twice — once against the FULL-vision prefix
    (v_full) and once against the all-masked prefix (v_masked) — then blend

        v = v_masked + w * (v_full - v_masked)

    w = 1 -> identical to stock sampling (vision fully trusted)
    w < 1 -> drift toward the language prior (less vision)
    w = 0 -> pure language prior

Cost: 2x denoise forwards per step, NO training.

This OFFLINE check (no simulator) verifies the mechanism is sound:
  * w = 1 reproduces stock ``sample_actions`` (correctness anchor)
  * as w: 1 -> 0 the action a(w) moves smoothly from a_full to a_masked
  * the blended velocity field stays well-scaled (no blow-up)

The same blend is the hook to later drop into ``robotwin_eval`` to measure OOD
success rate vs w (the real test of the hypothesis).

Usage
-----
    python scripts/step1_velocity_blend.py --ckpt ./ckpts/Hy-VLA-RoboTwin \
        --w 1.0 0.75 0.5 0.25 0.0 --out results/step1_blend.jsonl
"""

import argparse
import datetime
import json

import torch

from hy_vla import HyVLA, HyVLAConfig
from hy_vla.modeling_hy_vla import make_att_2d_masks

DEFAULT_INSTRUCTIONS = [
    "pick up the bottle",
    "open the drawer",
    "press the button",
]


def build_batch(instruction, image_keys, state_dim, device, dtype, k=6):
    img = torch.zeros(1, k, 3, 224, 224, device=device, dtype=dtype)
    batch = {key: img for key in image_keys}
    batch["observation.state"] = torch.zeros((1, state_dim), device=device, dtype=dtype)
    batch["task"] = [instruction]
    return batch


@torch.no_grad()
def build_prefix(model, images, img_masks, lang_tokens, lang_masks):
    """Replicate the prefix/KV-cache stage of ``sample_actions`` and return
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
def sample_blend(model, images, img_masks, lang_tokens, lang_masks, state, w, noise):
    """Euler-integrate the blended velocity field. Returns (action, traj) where
    traj[i] = (‖v_full‖, ‖v_masked‖, ‖v_blend‖) mean per-motor L2 at step i."""
    bsize = state.shape[0]
    device = state.device

    ppm_f, pkv_f = build_prefix(model, images, img_masks, lang_tokens, lang_masks)
    masked = [torch.zeros_like(m) for m in img_masks]
    ppm_m, pkv_m = build_prefix(model, images, masked, lang_tokens, lang_masks)

    dt = torch.tensor(-1.0 / model.config.num_steps, dtype=torch.float32, device=device)
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    def n(v):
        return v.detach().float().norm(dim=-1).mean().item()

    traj = []
    while time >= -dt / 2:
        et = time.expand(bsize)
        v_f, _ = model.denoise_step(state, ppm_f, pkv_f, x_t, et)
        v_m, _ = model.denoise_step(state, ppm_m, pkv_m, x_t, et)
        v = v_m + w * (v_f - v_m)
        traj.append((n(v_f), n(v_m), n(v)))
        x_t = x_t + dt * v
        time = time + dt
    return x_t, traj


def l2(a, b):
    return float((a - b).flatten().norm().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpts/Hy-VLA-RoboTwin")
    ap.add_argument("--out", default="results/step1_blend.jsonl")
    ap.add_argument("--w", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instructions", nargs="*", default=None)
    args = ap.parse_args()

    instructions = args.instructions or DEFAULT_INSTRUCTIONS
    device, dtype = "cuda", torch.bfloat16

    print(f"[load] {args.ckpt}")
    config = HyVLAConfig.from_pretrained(args.ckpt)
    policy = HyVLA.from_pretrained(args.ckpt, config=config)
    policy.enable_video_encoder_if_needed()
    policy = policy.to(device=device, dtype=dtype).eval()
    model = policy.model

    image_keys = list(config.image_features)
    act_dim = config.action_feature.shape[0]
    ws = sorted(set(args.w), reverse=True)  # 1.0 ... 0.0

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w")

    def emit(rec):
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    anchor_ok = True
    for instr in instructions:
        batch = build_batch(instr, image_keys, config.max_state_dim, device, dtype)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks, _ = policy.prepare_language(batch)

        torch.manual_seed(args.seed)
        noise = model.sample_noise((1, config.n_action_steps, config.max_action_dim), device)

        # correctness anchor: stock sampler with the same noise
        a_stock = model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(),
        )[..., :act_dim].float().cpu()

        actions = {}
        for w in ws:
            a, traj = sample_blend(model, images, img_masks, lang_tokens, lang_masks,
                                   state, w, noise)
            a = a[..., :act_dim].float().cpu()
            actions[w] = a
            emit({
                "type": "run", "instruction": instr, "w": w,
                "v_full_traj":  [t[0] for t in traj],
                "v_masked_traj": [t[1] for t in traj],
                "v_blend_traj":  [t[2] for t in traj],
                "action_abs_mean": float(a.abs().mean().item()),
                "action_std": float(a.std().item()),
                "action": a.squeeze(0).tolist(),
            })

        a_full = actions[ws[0]]   # w == max (1.0)
        a_lang = actions[ws[-1]]  # w == min (0.0)
        anchor = l2(actions[1.0], a_stock) if 1.0 in actions else float("nan")
        if 1.0 in actions and anchor > 1e-2:
            anchor_ok = False

        print(f"\n=== {instr!r} ===")
        if 1.0 in actions:
            print(f"  anchor ‖blend(w=1) - stock‖ = {anchor:.4f}  "
                  f"({'OK' if anchor <= 1e-2 else 'MISMATCH!'})")
        print(f"  {'w':>5} | {'‖a(w)-a_full‖':>13} | {'‖a(w)-a_lang‖':>13} | {'abs_mean':>8}")
        last_df = None
        mono = True
        for w in ws:
            df = l2(actions[w], a_full)
            dl = l2(actions[w], a_lang)
            am = float(actions[w].abs().mean().item())
            print(f"  {w:5.2f} | {df:13.4f} | {dl:13.4f} | {am:8.4f}")
            if last_df is not None and df < last_df - 1e-4:
                mono = False
            last_df = df
        print(f"  distance-from-full monotonic as w decreases: {mono}")

    summary = {
        "type": "summary",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ckpt": args.ckpt, "seed": args.seed, "w_grid": ws,
        "instructions": instructions,
        "anchor_w1_matches_stock": anchor_ok,
    }
    emit(summary)
    fout.close()
    print(f"\n[done] anchor_w1_matches_stock={anchor_ok}. wrote {args.out} — "
          f"download and send to Claude.")


if __name__ == "__main__":
    main()
