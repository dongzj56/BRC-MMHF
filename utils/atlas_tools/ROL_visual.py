#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overlay AAL ROI (41-42, bilateral hippocampus) on an MNI-aligned MRI,
then save as PNG & interactive HTML.
"""

import os, numpy as np, nibabel as nib
from nilearn import image, plotting

# -------------------------------------------------
# 1. Paths (Modify as needed)
# -------------------------------------------------
mri_path = rf"E:\adni_dataset\MRI\002_S_2043.nii"
aal_path = rf"E:\adni_dataset\AAL_space-MNI152NLin6_res-2x2x2\AAL.nii"
# aal_path = rf"E:\adni_dataset\aal_for_SPM8\ROI_MNI_V4.nii"

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
png_file  = os.path.join(output_dir, "mri_hippocampus_overlay.png")
html_file = os.path.join(output_dir, "mri_hippocampus_overlay.html")

# -------------------------------------------------
# 2. Read MRI & AAL atlas
# -------------------------------------------------
mri_img = nib.load(mri_path)
aal_img = nib.load(aal_path)
aal_data = aal_img.get_fdata()

# -------------------------------------------------
# 3. Generate Hippocampus ROI Mask (41 L-Hippocampus, 42 R-Hippocampus)
# -------------------------------------------------
roi_ids = [41,42]
mask_data = np.isin(aal_data, roi_ids).astype(np.uint8)
mask_img  = nib.Nifti1Image(mask_data, affine=aal_img.affine)

# If atlas and MRI resolution differ -> Resample mask to MRI grid
if mri_img.header.get_zooms() != aal_img.header.get_zooms():
    mask_img = image.resample_to_img(mask_img, mri_img, interpolation="nearest")

# -------------------------------------------------
# 4-A. Static PNG Overlay
# -------------------------------------------------
display = plotting.plot_roi(
    roi_img=mask_img,
    bg_img=mri_img,
    cmap="autumn",
    alpha=0.3,
    title="(red overlay)",
    draw_cross = False,  # Do not draw crosshairs
    annotate = False  # Do not show axes and scale
)
display.savefig(png_file, dpi=300)
display.close()
print(f"Static PNG saved: {png_file}")

# -------------------------------------------------
# 4-B. Interactive HTML Overlay
# -------------------------------------------------
view = plotting.view_img(
    mask_img,
    bg_img=mri_img,
    cmap="autumn",
    opacity=0.7,
    symmetric_cmap=False,
    title="Bilateral Hippocampus (interactive view)"
)
view.save_as_html(html_file)
print(f"Interactive HTML saved: {html_file}")
