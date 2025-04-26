# Data Directory

This directory contains or provides access to the ADHD-200 dataset used for the GNN-STAN model.

## ADHD-200 Dataset

The ADHD-200 dataset is a multi-site, open-access dataset containing resting-state fMRI data from 973 participants, including 362 individuals diagnosed with ADHD and 585 typically developing controls.

### Automatic Download

The easiest way to access the data is through the `nilearn` Python package, which provides a function to download a subset of the dataset:

```python
from nilearn import datasets
# Download data for 30 subjects (adjust as needed)
adhd_dataset = datasets.fetch_adhd(n_subjects=30)

# Access functional images
func_files = adhd_dataset.func

# Access phenotypic information including ADHD diagnosis
labels = adhd_dataset.phenotypic['adhd']  # 1 for ADHD, 0 for control
```

### Manual Download

To download the full ADHD-200 dataset:

1. Visit the official ADHD-200 Preprocessed repository: http://preprocessed-connectomes-project.org/adhd200/
2. Register for access if required
3. Download the preprocessed data
4. Place the downloaded data in the following directory structure:

```
data/
└── adhd200/
    ├── KKI/
    │   └── ...
    ├── NeuroIMAGE/
    │   └── ...
    ├── NYU/
    │   └── ...
    ├── OHSU/
    │   └── ...
    ├── Peking/
    │   └── ...
    ├── Pittsburgh/
    │   └── ...
    └── WashU/
        └── ...
```

## Dataset Structure

Each subject in the dataset typically has:

- **Anatomical MRI**: T1-weighted structural brain scans
- **Functional MRI**: Resting-state fMRI time series
- **Phenotypic data**: Information including:
  - Diagnosis (ADHD or control)
  - ADHD subtype (if applicable)
  - Age, gender, handedness
  - Various cognitive and behavioral measures

## Using Your Own Data

If you want to use your own rs-fMRI data:

1. Ensure your data is in NIfTI format (.nii or .nii.gz)
2. Create a consistent directory structure
3. Update the data loading functions in the preprocessing scripts accordingly

## Notes on Data Usage

- The preprocessing pipeline expects NIfTI-formatted rs-fMRI data
- Labels should be binary (1 for ADHD, 0 for control)
- Site information should be preserved for Leave-One-Site-Out (LOSO) cross-validation

## Data Privacy

When using the ADHD-200 dataset or any other medical imaging data:

- Follow all data usage agreements
- Do not attempt to re-identify subjects
- Use the data only for research purposes
- Cite the ADHD-200 Consortium in any publications using this data

## Citation

When using the ADHD-200 dataset, please cite:

```
ADHD-200 Consortium. (2012). The ADHD-200 Consortium: A Model to Advance the 
Translational Potential of Neuroimaging in Clinical Neuroscience. 
Frontiers in Systems Neuroscience, 6, 62. 
https://doi.org/10.3389/fnsys.2012.00062
```