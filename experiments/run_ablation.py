"""Unified CLI for main / module ablation / modality ablation experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from MMHF import Cfg, load_cfg, run_training


MODULE_VARIANTS = {
    "full": {
        "use_shared_layer": True,
        "use_channel_exchange": True,
        "use_hypergraph_conv": True,
        "use_hypergraph_attention": True,
        "use_mri": True,
        "use_pet": True,
        "use_table": True,
    },
    "no_hattn": {
        "use_hypergraph_attention": False,
    },
    "no_hgc": {
        "use_hypergraph_conv": False,
        "use_hypergraph_attention": False,
    },
    "no_ce": {
        "use_channel_exchange": False,
        "use_hypergraph_conv": False,
        "use_hypergraph_attention": False,
    },
    "no_sl": {
        "use_shared_layer": False,
        "use_channel_exchange": False,
        "use_hypergraph_conv": False,
        "use_hypergraph_attention": False,
    },
}

MODALITY_VARIANTS = {
    "mri": {"use_mri": True, "use_pet": False, "use_table": False},
    "pet": {"use_mri": False, "use_pet": True, "use_table": False},
    "table": {"use_mri": False, "use_pet": False, "use_table": True},
    "mri_pet": {"use_mri": True, "use_pet": True, "use_table": False},
    "mri_table": {"use_mri": True, "use_pet": False, "use_table": True},
    "pet_table": {"use_mri": False, "use_pet": True, "use_table": True},
    "mri_pet_table": {"use_mri": True, "use_pet": True, "use_table": True},
    "table36": {
        "use_mri": False,
        "use_pet": False,
        "use_table": True,
        "feature_schema": "common36",
    },
}


def load_cfg_dict_from_obj(cfg: Cfg) -> dict:
    out = {}
    for k, v in vars(cfg).items():
        if k == "device":
            out[k] = str(v)
        else:
            out[k] = v
    return out


def apply_variant(cfg: Cfg, experiment: str, variant: str) -> Cfg:
    base = load_cfg_dict_from_obj(cfg)
    if experiment == "main":
        variant = "full"
        overrides = dict(MODULE_VARIANTS["full"])
    elif experiment == "module_ablation":
        if variant not in MODULE_VARIANTS:
            raise ValueError(f"Unknown module variant: {variant}. Choose from {list(MODULE_VARIANTS)}")
        overrides = {**MODULE_VARIANTS["full"], **MODULE_VARIANTS[variant]}
    elif experiment == "modality_ablation":
        if variant not in MODALITY_VARIANTS:
            raise ValueError(f"Unknown modality variant: {variant}. Choose from {list(MODALITY_VARIANTS)}")
        overrides = {**MODULE_VARIANTS["full"], **MODALITY_VARIANTS[variant]}
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    base.update(overrides)
    base["experiment_name"] = f"{experiment}/{variant}"
    return Cfg(base)


def parse_folds(fold_arg: str, n_splits: int):
    if fold_arg.lower() == "all":
        return list(range(1, n_splits + 1))
    return [int(x) for x in fold_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="BRC-MMHF ablation / main experiment runner")
    parser.add_argument("--config", default="config/config2.json")
    parser.add_argument(
        "--experiment",
        default="main",
        choices=["main", "module_ablation", "modality_ablation"],
    )
    parser.add_argument(
        "--variant",
        default="full",
        help="full|no_hattn|no_hgc|no_ce|no_sl|mri|pet|table|mri_pet|mri_table|pet_table|mri_pet_table|table36",
    )
    parser.add_argument("--fold", default="all", help="all or fold id(s), e.g. 1 or 1,2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    raw = load_cfg(args.config)
    cfg = Cfg(raw)
    cfg = apply_variant(cfg, args.experiment, args.variant)

    if args.epochs is not None:
        cfg.num_epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(
            "checkpoints_ablation",
            args.experiment,
            args.variant if args.experiment != "main" else "full",
        )
    cfg.checkpoint_dir = out_dir
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # persist effective config
    eff_path = os.path.join(cfg.checkpoint_dir, "effective_config.json")
    with open(eff_path, "w", encoding="utf-8") as f:
        dump = load_cfg_dict_from_obj(cfg)
        json.dump(dump, f, indent=2)
    print(f"[Ablation] experiment={args.experiment} variant={args.variant}")
    print(f"[Ablation] checkpoint_dir={cfg.checkpoint_dir}")
    print(
        f"[Ablation] modalities mri={cfg.use_mri} pet={cfg.use_pet} table={cfg.use_table} "
        f"ce={cfg.use_channel_exchange} sl={cfg.use_shared_layer} "
        f"hgc={cfg.use_hypergraph_conv} hattn={cfg.use_hypergraph_attention}"
    )

    folds = parse_folds(args.fold, int(cfg.n_splits))
    run_training(cfg, folds=folds)


if __name__ == "__main__":
    main()
