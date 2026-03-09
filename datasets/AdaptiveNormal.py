"""
Purpose: Adaptive intensity normalization for medical images.
This function performs percentile-based normalization by:
- Filtering non-negative voxels
- Computing lower/upper percentile values
- Normalizing to zero mean and unit half-range
- Clamping the output to [-1, 1]
"""
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)
import nibabel as nib
import numpy as np
import torch
from os.path import join

import os


def adaptive_normal(img):  # Define a function named adaptive_normal that takes an image array as input
    min_p = 0.001  # Minimum percentile (controls lower bound of normalization)
    max_p = 0.999  # Maximum percentile (controls upper bound of normalization)

    imgArray = img  # Assign the input image to imgArray for subsequent processing
    imgPixel = imgArray[imgArray >= 0]  # Filter out pixel values below 0 (often invalid or background)
    imgPixel, _ = torch.sort(imgPixel)  # Sort filtered pixels; discard indices

    # Compute the index corresponding to the lower percentile value
    index = int(round(len(imgPixel) - 1) * min_p + 0.5)  # Position of the lower percentile
    if index < 0:  # Prevent out-of-range index
        index = 0
    if index > (len(imgPixel) - 1):  # Prevent out-of-range index
        index = len(imgPixel) - 1
    value_min = imgPixel[index]  # Lower percentile value

    # Compute the index corresponding to the upper percentile value
    index = int(round(len(imgPixel) - 1) * max_p + 0.5)  # Position of the upper percentile
    if index < 0:  # Prevent out-of-range index
        index = 0
    if index > (len(imgPixel) - 1):  # Prevent out-of-range index
        index = len(imgPixel) - 1
    value_max = imgPixel[index]  # Upper percentile value

    mean = (value_max + value_min) / 2.0  # Normalization mean: average of upper and lower values
    stddev = (value_max - value_min) / 2.0  # Normalization scale: half of the range

    imgArray = (imgArray - mean) / stddev  # Normalize the image to zero mean and unit half-range
    imgArray[imgArray < -1] = -1.0  # Clamp values below -1 to -1
    imgArray[imgArray > 1] = 1.0  # Clamp values above 1 to 1

    return imgArray  # Return the normalized image array
