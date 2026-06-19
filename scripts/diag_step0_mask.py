#!/usr/bin/env python
"""step 0 diagnostic — is masked-vision velocity a valid *language* prior?

Foundation check for the test-time-drift idea (see
``idea-test-time-drift-toward-language-manifold.md`` §6 step 0). Does NOT need
the RoboTwin simulator — just the checkpoint and one GPU.

Mechanism
---------
Hy-VLA conditions the action expert on vision + language via one concatenated
token sequence (shared attention, no CFG / no condition dropout, see idea md
§4.5). The only built-in lever to drop vision is the per-camera ``img_mask``:
when False, that camera's image *patch* tokens become padding and are dropped
from attention (``embed_prefix`` L1077-1078), leaving language (+state) to
drive the action. So "vision off" == ``img_masks = [zeros_like(m) ...]``.

We run, with the *same* sampling noise so differences are purely conditional:
  * NORMAL : img_masks as produced (vision on)
  * MASKED : all img_masks False (vision off -> language-only)
and monkeypatch ``denoise_step`` to log ``‖v_t‖`` at every Euler step.

Decisive question (falsification): across different instructions, does the
MASKED action chunk change? If masked outputs are ~identical regardless of the
instruction, there is no language manifold to drift toward and the idea is
blocked on this base model.

Usage
-----
    python scripts/diag_step0_mask.py --ckpt ./ckpts/Hy-VLA-RoboTwin \
        --out step0_mask_diag.jsonl

One jsonl record per (instruction, branch); a final ``summary`` record holds
the cross-instruction sensitivity numbers. Print also goes to stdout.
"""

import argparse
import datetime
import itertools
import json

import torch

from hy_vla import HyVLA, HyVLAConfig

DEFAULT_INSTRUCTIONS = [
    "",  # near-unconditional reference
    "pick up the bottle",
    "open the drawer",
    "stack the blocks",
    "press the button",
    "hand over the cup",
]


def build_batch(instruction, image_keys, state_dim, device, dtype, k=6):
    """Dummy zero observation + one instruction. Vision content is irrelevant
    for the masked branch; for the normal branch this is a degenerate (zero)
    image but still exercises the full conditioning pathway."""
    img = torch.zeros(1, k, 3, 224, 224, device=device, dtype=dtype)
    batch = {key: img for key in image_keys}
    batch["observation.state"] = torch.zeros((1, state_dim), device=device, dtype=dtype)
    batch["task"] = [instruction]
    return batch


def run_branch(policy, batch, mask_vision, noise):
    """Returns (action[1,T,D] float cpu, list_of_per_step_velocity_norms)."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks, _ = policy.prepare_language(batch)
    if mask_vision:
        img_masks = [torch.zeros_like(m) for m in img_masks]

    v_norms = []  # mean per-motor L2 norm of v_t, one entry per Euler step
    model = policy.model
    orig = model.denoise_step

    def wrapped(state_, prefix_pad_masks, past_key_values, x_t, timestep):
        v_t, att = orig(state_, prefix_pad_masks, past_key_values, x_t, timestep)
        # v_t: (B, n_action_steps, action_dim) -> L2 over action_dim, mean over (B, steps)
        v_norms.append(v_t.detach().float().norm(dim=-1).mean().item())
        return v_t, att

    model.denoise_step = wrapped
    try:
        with torch.no_grad():
            a = model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise.clone(), vis_attn=False,
            )
    finally:
        model.denoise_step = orig
    return a.detach().float().cpu(), v_norms


def l2(a, b):
    return float((a - b).flatten().norm().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpts/Hy-VLA-RoboTwin")
    ap.add_argument("--out", default="step0_mask_diag.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instructions", nargs="*", default=None,
                    help="override the default instruction set")
    args = ap.parse_args()

    instructions = args.instructions or DEFAULT_INSTRUCTIONS
    device = "cuda"
    dtype = torch.bfloat16

    print(f"[load] {args.ckpt}")
    config = HyVLAConfig.from_pretrained(args.ckpt)
    policy = HyVLA.from_pretrained(args.ckpt, config=config)
    policy.enable_video_encoder_if_needed()
    policy = policy.to(device=device, dtype=dtype).eval()

    image_keys = list(config.image_features)
    act_dim = config.action_feature.shape[0]
    print(f"[cfg] image_keys={image_keys} n_action_steps={config.n_action_steps} "
          f"max_action_dim={config.max_action_dim} action_feature_dim={act_dim}")

    # One fixed noise reused for EVERY (instruction, branch) so all differences
    # are purely from conditioning, not the random start.
    torch.manual_seed(args.seed)
    noise = policy.model.sample_noise(
        (1, config.n_action_steps, config.max_action_dim), device
    )

    fout = open(args.out, "w")

    def emit(rec):
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    normal_actions, masked_actions = {}, {}
    print("\n  instruction                     | ‖a_normal-a_masked‖ (vision displacement)")
    print("  " + "-" * 72)
    for instr in instructions:
        batch = build_batch(instr, image_keys, config.max_state_dim, device, dtype)

        a_norm, vn_norm = run_branch(policy, batch, mask_vision=False, noise=noise)
        a_mask, vn_mask = run_branch(policy, batch, mask_vision=True, noise=noise)

        a_norm_t = a_norm[..., :act_dim]
        a_mask_t = a_mask[..., :act_dim]
        normal_actions[instr] = a_norm_t
        masked_actions[instr] = a_mask_t

        disp = l2(a_norm_t, a_mask_t)
        print(f"  {instr[:30]:<30} | {disp:.4f}")

        for branch, a_t, vn in (("normal", a_norm_t, vn_norm), ("masked", a_mask_t, vn_mask)):
            emit({
                "type": "run",
                "instruction": instr,
                "branch": branch,
                "vision_displacement": disp,          # same for both, convenience
                "v_norm_traj": vn,                     # ‖v_t‖ per Euler step
                "action_abs_mean": float(a_t.abs().mean().item()),
                "action_std": float(a_t.std().item()),
                "action": a_t.squeeze(0).tolist(),     # (T, act_dim)
            })

    # ---- cross-instruction sensitivity: the decisive numbers ----
    def mean_pairwise(actions_by_instr):
        keys = [k for k in actions_by_instr if k != ""]  # exclude empty ref
        ds = [l2(actions_by_instr[a], actions_by_instr[b])
              for a, b in itertools.combinations(keys, 2)]
        return (sum(ds) / len(ds)) if ds else 0.0

    masked_sens = mean_pairwise(masked_actions)   # >0 means language moves masked action
    normal_sens = mean_pairwise(normal_actions)   # reference scale
    ratio = (masked_sens / normal_sens) if normal_sens > 1e-9 else 0.0

    summary = {
        "type": "summary",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ckpt": args.ckpt,
        "seed": args.seed,
        "instructions": instructions,
        "masked_instruction_sensitivity": masked_sens,
        "normal_instruction_sensitivity": normal_sens,
        "masked_over_normal_ratio": ratio,
    }
    emit(summary)
    fout.close()

    print("\n=== step 0 verdict ===")
    print(f"  masked-branch instruction sensitivity : {masked_sens:.4f}")
    print(f"  normal-branch instruction sensitivity : {normal_sens:.4f}")
    print(f"  ratio (masked / normal)               : {ratio:.3f}")
    if masked_sens < 1e-3 or ratio < 0.05:
        print("  -> masked action barely depends on the instruction: NO usable")
        print("     language manifold. Idea blocked on this base model (needs a")
        print("     vision-dropout finetune to instill an unconditional branch).")
    else:
        print("  -> masked action DOES track the instruction: a language prior")
        print("     exists. §2.1 holds; proceed to (b) latent drift.")
    print(f"\n[done] wrote {args.out} — download it and send to Claude for analysis.")


if __name__ == "__main__":
    main()
