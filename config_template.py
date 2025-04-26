"""
Configuration template for the GNN-STAN ADHD classification project.

Instructions:
1. Copy this file to 'config.py'
2. Fill in your actual credentials and configuration values
3. Make sure to keep 'config.py' in your .gitignore file
"""

# Dataset configuration
DATASET_CONFIG = {
    # Path to the ADHD-200 dataset
    "adhd200_path": "/path/to/adhd200/dataset",
    
    # Alternative: Credentials for dataset API access (if applicable)
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here"
}

# Computation resources
COMPUTE_CONFIG = {
    # GPU settings
    "use_gpu": True,
    "gpu_id": 0,
    
    # Parallel processing
    "num_workers": 4,
    
    # Memory limits (in MB)
    "memory_limit": 16000,
}

# Output directories
OUTPUT_CONFIG = {
    "preprocessed_dir": "preprocessed_data",
    "results_dir": "results",
    "models_dir": "saved_models",
    "logs_dir": "logs",
}

# Default model parameters (can be overridden by command line arguments)
MODEL_DEFAULTS = {
    "hidden_channels": 64,
    "gnn_layers": 3,
    "gnn_type": "gcn",
    "dropout": 0.1,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "epochs": 100,
    "batch_size": 32,
    "use_dynamic": True,
    "bidirectional": True,
}

# Experiment settings
EXPERIMENT_CONFIG = {
    "random_seed": 42,
    "cross_validation_folds": 5,
    
    # For hyperparameter tuning
    "search_type": "grid",  # "grid" or "random"
    "n_random_configs": 20,
    "n_splits": 5,
    
    # For LOSO-CV
    "include_sites": ["NYU", "KKI", "OHSU", "Peking", "Pittsburgh", "WashU", "NeuroIMAGE", "Brown"],
}