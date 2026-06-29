"""
Minimal DynamicVGGT inference example (code-only release, no bundled weights).

Provide your own checkpoint via --checkpoint, or initialize randomly for debugging.
"""

import argparse

import torch

from dynamicvggt.models.dynamicvggt import DynamicVGGT


def build_dummy_multiview_batch(batch_size=1, time_steps=2, views=2, height=518, width=518, device="cpu"):
    images = torch.rand(batch_size, time_steps, views, 3, height, width, device=device)
    return {
        "input_images": images,
        "input_extrinsics": torch.eye(4, device=device).view(1, 1, 1, 4, 4).expand(
            batch_size, time_steps, views, 4, 4
        )[..., :3, :],
        "input_intrinsics": torch.tensor(
            [[[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]]],
            device=device,
        ).view(1, 1, 1, 3, 3).expand(batch_size, time_steps, views, 3, 3),
        "has_flow": torch.tensor([False], device=device),
        "has_moge_depth": torch.tensor([False], device=device),
        "time_gap": 1,
    }


def main():
    parser = argparse.ArgumentParser(description="DynamicVGGT forward-pass example")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a user-provided checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-gs", action="store_true", help="Enable Gaussian-splatting heads (needs gsplat)")
    args = parser.parse_args()

    device = args.device
    model = DynamicVGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_gs=args.enable_gs,
        enable_future_point=True,
    ).to(device)
    model.eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")

    batch = build_dummy_multiview_batch(device=device)
    with torch.no_grad():
        outputs = model(batch)

    print("Forward pass OK. Output keys:", sorted(outputs.keys()))


if __name__ == "__main__":
    main()
