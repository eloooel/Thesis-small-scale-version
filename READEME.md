# GNN-STAN for ADHD Classification

A hybrid Graph Neural Network and Spatio-Temporal Attention Network model for classifying ADHD from rs-fMRI data.

## Overview

This repository contains the implementation of our thesis work "Enhanced ADHD Diagnosis Using Hybrid Deep Learning: An rs-fMRI Analysis". The GNN-STAN model combines Graph Neural Networks for modeling brain connectivity patterns with Spatio-Temporal Attention Networks for identifying important brain regions and time periods.

The model is designed to analyze resting-state functional MRI (rs-fMRI) data from the ADHD-200 dataset to classify subjects as either having ADHD or being typically developing controls.

## Repository Structure

```
gnn-stan-adhd/
├── data/                      # Data directory
│   └── README.md              # Instructions for data setup
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── gnn.py             # Graph Neural Network component
│   │   ├── stan.py            # Spatio-Temporal Attention Network component
│   │   └── gnn_stan.py        # Hybrid model implementation
│   ├── preprocessing/         # Data preprocessing
│   │   ├── preprocessing.py   # Main preprocessing pipeline
│   │   ├── feature_extraction.py  # Feature extraction functions
│   │   └── brain_graph.py     # Brain graph creation
│   └── utils/                 # Utility functions
│       ├── visualization.py   # Visualization functions
│       └── metrics.py         # Evaluation metrics
├── experiments/               # Experiment scripts
│   ├── mini_test.py           # Small-scale test script
│   ├── loso_cv.py             # Leave-One-Site-Out cross-validation
│   └── hyperparameter_tuning.py  # Hyperparameter tuning
├── notebooks/                 # Jupyter notebooks
│   └── gnn_stan_mini_test.ipynb  # Notebook version of small-scale test
├── results/                   # Results directory
├── .gitignore                 # Git ignore file
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gnn-stan-adhd.git
cd gnn-stan-adhd
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the data directory (see `data/README.md` for details).

## Usage

### Small-Scale Test

To run the small-scale test with 2 patients (automatically downloads sample data):

```bash
python experiments/mini_test.py
```

This script will:
1. Download 2 subjects from the ADHD-200 dataset
2. Preprocess the rs-fMRI data
3. Extract features and create brain graphs
4. Train and evaluate a small GNN-STAN model
5. Analyze resource usage and make recommendations for scaling up

### Using VS Code

If you're using Visual Studio Code:

1. Open the repository folder in VS Code
2. Make sure you have the Python extension installed
3. Select your virtual environment as the Python interpreter
4. Open any Python file and run it using the play button or right-click menu

### Using Jupyter Notebook

To run the Jupyter notebook version:

```bash
jupyter notebook notebooks/gnn_stan_mini_test.ipynb
```

## Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Motion correction**: Compensates for head movement during scanning
2. **Slice timing correction**: Adjusts for temporal differences in slice acquisition
3. **Spatial normalization**: Warps individual brains to MNI152 standard space
4. **Temporal filtering**: Removes physiological noise and scanner-related artifacts

## Model Architecture

The GNN-STAN model consists of:

1. **GNN Component**: Models the brain as a graph where nodes are brain regions and edges represent functional connectivity
2. **STAN Component**: Applies spatial attention to identify important brain regions and temporal attention to focus on relevant time points
3. **Classification Layer**: Makes the final ADHD/control prediction

## Evaluation

The model uses several evaluation strategies:

- **Leave-One-Site-Out (LOSO) Cross-Validation**: Trains on data from all sites except one, tests on the left-out site
- **Stratified K-Fold Cross-Validation**: Ensures balanced representation of ADHD and control subjects in each fold
- **Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC

## Citation

If you use this code in your research, please cite our work:

```
@article{dacayo2025enhanced,
  title={Enhanced ADHD Diagnosis Using Hybrid Deep Learning: An rs-fMRI Analysis},
  author={Dacayo, Raphael Angelo F. and Dizon, Eli Aleandro M. and Immaculata, Allen Gregg M.},
  journal={University of Santo Tomas},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ADHD-200 Consortium for providing the dataset
- University of Santo Tomas, College of Information and Computing Sciences
- Our adviser, Maria Amelia Damian