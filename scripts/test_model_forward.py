"""CPU-friendly forward smoke test for BRCMMHF variants (no training)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.brc_mmhf import BRCMMHF


VARIANTS = {
    "full": dict(
        use_shared_layer=True,
        use_channel_exchange=True,
        use_hypergraph_conv=True,
        use_hypergraph_attention=True,
        use_mri=True,
        use_pet=True,
        use_table=True,
    ),
    "no_hattn": dict(use_hypergraph_attention=False),
    "no_hgc": dict(use_hypergraph_conv=False, use_hypergraph_attention=False),
    "no_ce": dict(
        use_channel_exchange=False,
        use_hypergraph_conv=False,
        use_hypergraph_attention=False,
    ),
    "no_sl": dict(
        use_shared_layer=False,
        use_channel_exchange=False,
        use_hypergraph_conv=False,
        use_hypergraph_attention=False,
    ),
    "mri": dict(use_mri=True, use_pet=False, use_table=False),
    "pet": dict(use_mri=False, use_pet=True, use_table=False),
    "table": dict(use_mri=False, use_pet=False, use_table=True),
    "mri_pet": dict(use_mri=True, use_pet=True, use_table=False),
    "mri_table": dict(use_mri=True, use_pet=False, use_table=True),
    "pet_table": dict(use_mri=False, use_pet=True, use_table=True),
    "mri_pet_table": dict(use_mri=True, use_pet=True, use_table=True),
    "table36": dict(use_mri=False, use_pet=False, use_table=True, feature_schema="common36"),
}


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def small_unet_kwargs(use_small: bool):
    if use_small:
        # UpConv halves channels: bottleneck must be 2 * level_channels[2]
        return dict(level_channels=[8, 16, 32], bottleneck_channel=64, roi_feat_dim=8)
    return dict(level_channels=[64, 128, 256], bottleneck_channel=512, roi_feat_dim=64)


def run_variant(name: str, base_flags: dict, cfg: dict, aal_path: str, use_small: bool, device: torch.device):
    flags = {**VARIANTS["full"], **VARIANTS[name]}

    tab_dim = 36 if flags.get("feature_schema") == "common36" else int(cfg.get("tab_dim") or 133)
    kw = small_unet_kwargs(use_small)
    # When using small U-Net, decoder out channels = level_channels[0]; ROI feat dim must match
    roi_c = kw["level_channels"][0]
    kw["roi_feat_dim"] = roi_c

    need_aal = flags["use_mri"] or flags["use_pet"]
    model = BRCMMHF(
        tab_dim=tab_dim,
        num_classes=2,
        aal_path=aal_path if need_aal else "",
        roi_range=(int(cfg.get("roi_start", 1)), int(cfg.get("roi_end", 90))),
        image_feature_dim=int(cfg.get("image_feature_dim", 128)),
        tab_feature_dim=int(cfg.get("tab_feature_dim", 64)),
        use_seg_task=False,  # faster smoke
        ks=tuple(cfg.get("hg_ks", [6, 18])),
        **kw,
        **{k: flags[k] for k in (
            "use_mri", "use_pet", "use_table",
            "use_shared_layer", "use_channel_exchange",
            "use_hypergraph_conv", "use_hypergraph_attention",
        )},
    ).to(device)
    model.eval()

    b = 2
    # tiny spatial size for CPU speed when small config
    # Spatial dims must be divisible by 8 for 3-level U-Net pool/upsample alignment
    spatial = (16, 24, 16) if use_small else (96, 112, 96)
    mri = torch.randn(b, 1, *spatial, device=device) if flags["use_mri"] else None
    pet = torch.randn(b, 1, *spatial, device=device) if flags["use_pet"] else None
    tab = torch.randn(b, tab_dim, device=device) if flags["use_table"] else None

    # For small spatial, ROI atlas may be larger — ROIPooling crops feat to atlas DHW.
    # If using small U-Net with tiny volumes, skip real atlas: replace pool with identity-like stub.
    if use_small and need_aal and model.pool is not None:
        n_roi = int(cfg.get("roi_end", 90)) - int(cfg.get("roi_start", 1)) + 1

        class _FakePool(torch.nn.Module):
            def forward(self, feat):
                # feat [B,C,D,H,W] -> [B,N,C]
                return feat.mean(dim=(2, 3, 4)).unsqueeze(1).expand(-1, n_roi, -1).contiguous()

        model.pool = _FakePool()

    with torch.no_grad():
        out = model(mri=mri, pet=pet, tabular=tab)
    logits = out["logits"]
    assert logits.shape == (b, 2), f"{name}: expected {(b, 2)}, got {tuple(logits.shape)}"
    msg = f"[OK] {name:14s} logits={tuple(logits.shape)}"
    lans = out.get("lans_scores")
    if lans is not None:
        msg += f" lans={tuple(lans.shape)}"
    print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config2.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--small",
        action="store_true",
        default=True,
        help="Use tiny U-Net channels / spatial size (default True for CPU)",
    )
    parser.add_argument("--full-size", action="store_true", help="Use paper-sized tensors (slow on CPU)")
    parser.add_argument(
        "--variants",
        default="full,no_hattn,no_hgc,no_ce,no_sl,mri,pet,table,mri_pet,mri_table,pet_table,mri_pet_table,table36",
    )
    args = parser.parse_args()
    use_small = not args.full_size

    cfg = load_cfg(args.config)
    aal = cfg.get("AAL_dir", "")
    if not use_small and (not aal or not os.path.isfile(aal)):
        raise FileNotFoundError(f"AAL atlas not found: {aal}")

    device = torch.device(args.device)
    names = [x.strip() for x in args.variants.split(",") if x.strip()]
    print(f"device={device} small={use_small} variants={names}")
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name}")
        run_variant(name, VARIANTS[name], cfg, aal, use_small, device)
    print("All forward smoke tests passed.")


if __name__ == "__main__":
    main()
