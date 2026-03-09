#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atlas Query Tool
----------------
1. Read label map (AAL / Hammers etc.) NIfTI + JSON LUT
2. Interactive Visualization
3. Support:
   • voxel -> ROI name
   • world -> nearest ROI centroid + actual label
"""

import numpy as np
import nibabel as nib
import torch
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import json
from pathlib import Path

# -------------------------------------------------
# 1) Read JSON LUT (NeuroParc / BIDS style)
# -------------------------------------------------
def load_aal_json_lut(json_path: str, return_center=False, return_size=False):
    """
    Parse JSON
    ----------
    Parameters
    ----------
    json_path : str
        JSON file path
    return_center , return_size : bool
        Whether to return centroid coordinates / ROI voxel count

    Returns
    -------
    lut : dict[int, str]
        Label ID -> Name
    centers : dict[int, tuple]  (Optional)
    sizes   : dict[int, int]    (Optional)
    """
    p = Path(json_path)
    with open(p, "r", encoding="utf-8") as f:
        js = json.load(f)

    lut, centers, sizes = {}, {}, {}
    for k, v in js["rois"].items():
        idx = int(k)
        if idx == 0 or v["label"] in (None, "null"):
            continue                  # Skip background or empty ROI
        lut[idx]      = v["label"]
        centers[idx]  = tuple(v["center"]) if v["center"] else None
        sizes[idx]    = v["size"]

    if return_center or return_size:
        return lut, centers, sizes
    return lut


# -------------------------------------------------
# 2) Nearest ROI Centroid Mapping
# -------------------------------------------------
def nearest_roi(world_xyz, centers: dict):
    """
    Given MNI coordinates -> find nearest ROI centroid
    Filter ROI with center is None to avoid TypeError
    """
    w = np.asarray(world_xyz)
    # Calculate Euclidean distance only for valid centroids
    valid = ((k, np.asarray(c)) for k, c in centers.items() if c is not None)
    lab, dist = min(((k, np.linalg.norm(w - c)) for k, c in valid),
                    key=lambda t: t[1])
    return lab, dist


# -------------------------------------------------
# 3) Main Program
# -------------------------------------------------
if __name__ == "__main__":
    # ======= Modify to your local actual path =======
    atlas_nii = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.nii"
    lut_file  = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.json"
    # =====================================

    bg_template = load_mni152_template(resolution=2)

    # ---------- Load Atlas ----------
    img  = nib.load(atlas_nii)
    data = img.get_fdata().astype(int)
    torch_data = torch.from_numpy(data).long()   # Available for PyTorch processing if needed

    lut, centers, sizes = load_aal_json_lut(lut_file,
                                            return_center=True,
                                            return_size=True)

    labels = np.unique(data)
    print(f"Total Labels: {labels.size} (including 0 background)")
    print(f"Max Label ID: {labels.max()}")
    print(f"Atlas Voxel Grid: {img.shape}, Voxel Size: {np.abs(np.diag(img.affine)[:3])} mm")

    # ---------- Interactive View ----------
    view = plotting.view_img(
        atlas_nii,
        bg_img=bg_template,
        cmap="tab20",
        threshold=0,
        opacity=0.55,
        title="AAL Atlas (2 mm)"
    )
    try:
        view.open_in_browser()
    except RuntimeError:
        out_html = "/data/coding/Multimodal_AD/output/atlas_view.html"
        view.save_as_html(out_html)
        print(f"No GUI environment, interactive HTML saved to: {out_html}")

    # -------------------------------------------------
    # 4) Query Functions
    # -------------------------------------------------
    def query_voxel(i, j, k):
        """Voxel index -> ROI Name"""
        if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
            print("Index out of bounds")
            return
        val = int(data[i, j, k])
        print(f"[Voxel] ({i},{j},{k}) -> label {val}: {lut.get(val, 'Background/Unknown')}")
        return val

    def query_world(x, y, z):
        """
        World Coordinate Query
        1. Output the label of the point
        2. Find nearest ROI centroid and show distance, name etc.
        """
        world = (x, y, z)

        # 1) Actual Label
        ijk = np.round(np.linalg.inv(img.affine) @ [*world, 1])[:3].astype(int)
        true_lab = None
        if (ijk >= 0).all() and (ijk < data.shape).all():
            true_lab = int(data[tuple(ijk)])

        # 2) Nearest Centroid
        lab_cen, dist = nearest_roi(world, centers)
        cen_xyz = centers[lab_cen]
        size    = sizes[lab_cen]
        name    = lut[lab_cen]

        # ---------- Output ----------
        print("\n=== World Query Result ===")
        print(f"Input Coordinates   : ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        print(f"Voxel Index         : {tuple(ijk)}")
        print(f"Voxel Label         : {true_lab} ({lut.get(true_lab, 'Background/Unknown')})")
        print("-- Nearest ROI Centroid --")
        print(f"ROI ID              : {lab_cen}")
        print(f"ROI Name            : {name}")
        print(f"ROI Centroid (mm)   : ({cen_xyz[0]:.1f}, {cen_xyz[1]:.1f}, {cen_xyz[2]:.1f})")
        print(f"Distance to Centroid: {dist:.2f} mm")
        print(f"ROI Voxel Count     : {size}")
        print("========================\n")
        return lab_cen, dist

    # -------------------------------------------------
    # 5) Demo
    # -------------------------------------------------
    print("\n######### DEMO #########")
    query_voxel(75, 97, 50)
    query_world(-34, -20, -18)   # Old reference coordinates for Left Hippocampus
    query_world(-27, -18, -24)   # AAL Left Hippocampus centroid
    query_world(11, -80, 24)     # Occipital Lobe example
