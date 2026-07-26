#!/usr/bin/env python
"""Generate TabPFN table embeddings consumed by MMHF.py."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.tabpfn_embedding import build_tabpfn_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TabPFN tabular embeddings")
    parser.add_argument("--table_csv", default="adni_dataset/ADNI_Tabel.csv")
    parser.add_argument("--out_csv", default="adni_dataset/tabular_embeddings.csv")
    parser.add_argument("--label_col", default="Group")
    parser.add_argument("--classes", default="SMCI,PMCI")
    parser.add_argument("--start_col", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dropna", action="store_true")
    args = parser.parse_args()

    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    df = build_tabpfn_embeddings(
        args.table_csv,
        args.out_csv,
        label_col=args.label_col,
        classes=classes,
        start_col=args.start_col,
        device=args.device,
        dropna=args.dropna,
    )
    print(f"[OK] saved {args.out_csv} shape={df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
