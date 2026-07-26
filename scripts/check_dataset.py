#!/usr/bin/env python
"""Integrity check CLI for ADNI multimodal data."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.checks import main

if __name__ == "__main__":
    raise SystemExit(main())
