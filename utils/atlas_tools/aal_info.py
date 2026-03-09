import nibabel as nib
import numpy as np

# ---------- 1. Load AAL Template ----------
aal_path = rf"E:\adni_dataset\AAL_space-MNI152NLin6_res-2x2x2.nii\AAL_space-MNI152NLin6_res-2x2x2.nii"
img   = nib.load(aal_path)
data  = img.get_fdata()          # Floating point ndarray
total = data.size

# ---------- 2. Define Three Masks ----------
mask0        = np.isclose(data, 0.0)
mask_1_94    = np.logical_and(data >= 1,  data <= 94)    # 1-94
mask_95_120  = np.logical_and(data >= 95, data <= 120)   # 95-120

cnt0        = np.count_nonzero(mask0)
cnt_1_94    = np.count_nonzero(mask_1_94)
cnt_95_120  = np.count_nonzero(mask_95_120)

# ---------- 3. Output ----------
print(f"Total Voxels : {total}")
print(f"Label 0   Voxels : {cnt0}   ({cnt0/total:.4%})")
print(f"Label 1-94 Voxels : {cnt_1_94}   ({cnt_1_94/total:.4%})")
print(f"Label 95-120 Voxels : {cnt_95_120}   ({cnt_95_120/total:.4%})")
