# BRC-MMHF
**Brain Region-Centered MultiModal Hypergraph Fusion for MCI Conversion Prediction**

## Overview
BRC-MMHF is a multimodal deep learning framework designed to predict the conversion of Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD). By integrating MRI and PET imaging data with clinical tabular data, the model leverages a novel Brain Region-Centered hypergraph fusion mechanism to capture complex, high-order correlations between brain regions and modalities.

## Key Features
- **Multimodal Hypergraph Fusion Framework**: A novel framework for MCI conversion prediction that addresses cross-modal alignment and high-order modeling challenges in MRI-PET fusion.
- **Dual-Stream 3D U-Net with CEN**: Utilizes a parameter-free channel exchange mechanism and ROI pooling to reduce modality heterogeneity and extract structurally consistent features from MRI and PET images.
- **Structured Clinical Data Integration**: Integrates clinical data through a lightweight tabular encoder to improve model adaptability and diagnostic robustness.
- **Interpretability Analysis**: Incorporates a lesion-region identification module to highlight disease-relevant brain regions, enhancing clinical interpretability alongside tabular variable analysis.

## Requirements
The project requires Python 3.10+ and the following major dependencies:
- `torch==2.4.1`
- `monai==1.4.0`
- `pandas==2.2.3`
- `scikit-learn==1.6.1`
- `numpy==1.26.4`
- `nibabel==5.3.2`
- `SimpleITK==2.5.0`
- `nilearn==0.11.1`

For a full list of dependencies, refer to [requirements.txt](env/requirements.txt).

## Data Preparation
This framework is designed to work with the ADNI dataset and validates on the SCAN dataset.

**Datasets**:
- **ADNI**: The [Alzheimer's Disease Neuroimaging Initiative](https://ida.loni.usc.edu/) dataset is used for model development. Access applications are available via the LONI Image & Data Archive.
- **SCAN**: The Standardized Centralized Alzheimer’s Neuroimaging (SCAN) dataset serves as an external validation set. SCAN is a subset of the **National Alzheimer’s Coordinating Center (NACC)**.

**Processing Pipeline**:
- **Preprocessing**: The preprocessing pipeline for ADNI data is available here: [adni_image_process](https://github.com/dongzj56/adni_image_process.git)
- **Data Screening**: Tools for data filtering and subject selection: [ADNI_data_filter_code](https://github.com/dongzj56/ADNI_data_filter_code.git)
- **Reference**: The preprocessing and filtering workflow references the [Clinica tool](https://github.com/aramis-lab/clinica.git).

## Contact
Author: Z.J. Dong

Email: dongzj56@gmail.com
