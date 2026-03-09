#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

def compute_p_values(arr1, arr2):
    x = np.asarray(arr1, dtype=float)
    y = np.asarray(arr2, dtype=float)
    if x.shape != y.shape:
        raise ValueError("The two arrays must have the same length!")

    t_stat, p_t = stats.ttest_rel(x, y)

    try:
        w_stat, p_w = stats.wilcoxon(x, y, correction=True)
    except ValueError as e:
        p_w = np.nan
        print("Wilcoxon test cannot be performed: ", e)

    return {"t-test": p_t, "wilcoxon": p_w}

if __name__ == "__main__":
    model_a = [0.9152, 0.8830, 0.9218, 0.9340, 0.9418]
    model_b = [0.9867, 0.9767, 0.9806, 0.9845, 0.9751]

    p_vals = compute_p_values(model_a, model_b)
    print("Paired t-test p-value: ", p_vals["t-test"])
    print("Wilcoxon p-value: ", p_vals["wilcoxon"])
