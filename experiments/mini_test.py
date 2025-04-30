"""
Enhanced small-scale test of the GNN-STAN model for ADHD classification.

This script demonstrates the complete pipeline with proper evaluation metrics,
baseline comparison, and cross-validation to provide a stronger proof of concept.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import psutil
from tqdm import tqdm
import glob
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.preprocessing.preprocessing import Preprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.brain_graph import BrainGraphCreator
from src.models.gnn_stan import GNN_STAN_Classifier
from src.utils.metrics import calculate_binary_metrics, print_metrics_summary
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve

def load_local_peking_data(data_dir, max_subjects=6):
    """
    Load the local Peking_1 dataset (limited to max_subjects)
    
    Parameters:
    -----------
    data_dir : str
        Path to the Peking_1 dataset directory
    max_subjects : int
        Maximum number of subjects to load
    
    Returns:
    --------
    dict
        Dictionary containing functional files, subject IDs, and labels
    """
    print(f"Loading local Peking_1 dataset from {data_dir}")
    
    # Find all functional MRI files
    func_pattern = os.path.join(data_dir, "sub-*", "ses-*", "func", "*_task-rest_*.nii.gz")
    all_func_files = sorted(glob.glob(func_pattern))
    
    if not all_func_files:
        raise FileNotFoundError(f"No functional MRI files found in {data_dir}")
    
    # Limit to max_subjects
    func_files = all_func_files[:max_subjects]
    
    # Extract subject IDs from filenames
    subject_ids = []
    for func_file in func_files:
        # Extract subject ID from the file path (sub-XXXXXXX)
        parts = os.path.normpath(func_file).split(os.sep)
        for part in parts:
            if part.startswith("sub-"):
                subject_ids.append(part)
                break
    
    # For demonstration purposes, we'll use a simple rule for ADHD labels:
    # Odd numbered subjects = ADHD, Even numbered subjects = Control
    # You should replace this with actual diagnostic information if available
    labels = []
    for subject_id in subject_ids:
        # Extract the numeric part (remove 'sub-' prefix)
        subject_num = subject_id.replace('sub-', '')
        try:
            # If last digit is odd, classify as ADHD (1), else Control (0)
            if int(subject_num[-1]) % 2 == 1:
                labels.append(1)  # ADHD
            else:
                labels.append(0)  # Control
        except:
            # Default to Control if conversion fails
            labels.append(0)
    
    print(f"Found {len(func_files)} subjects (limited to {max_subjects}):")
    for i, (subject, label) in enumerate(zip(subject_ids, labels)):
        print(f"  {i+1}. {subject}: {'ADHD' if label == 1 else 'Control'}")
    
    return {
        'func': func_files,
        'subject_ids': subject_ids,
        'labels': labels
    }

def measure_resource_usage(func):
    """Decorator to measure time and memory usage of a function"""
    def wrapper(*args, **kwargs):
        # Start measurements
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Call the function
        result = func(*args, **kwargs)
        
        # End measurements
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Print results
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        
        return result
    return wrapper

@measure_resource_usage
def run_preprocessing_pipeline(func_files, subject_ids):
    """Run the preprocessing pipeline on the functional files"""
    # Initialize preprocessor
    preprocessor = Preprocessor(output_dir="preprocessed_data")
    
    # Preprocess the data
    results = preprocessor.batch_preprocess(func_files, subject_ids)
    
    return results

@measure_resource_usage
def extract_features(preprocessed_data):
    """Extract features from the preprocessed data"""
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        connectivity_type='correlation',
        threshold=0.2,
        dynamic=True,  # Enable dynamic connectivity for STAN
        window_size=20,
        step_size=5
    )
    
    # Extract features
    features = feature_extractor.batch_extract_features(preprocessed_data)
    
    return features

@measure_resource_usage
def create_brain_graphs(features):
    """Create brain graphs from the extracted features"""
    # Initialize brain graph creator
    graph_creator = BrainGraphCreator(
        threshold=0.2,
        use_absolute=True,
        self_loops=False,
        use_dynamic=True  # Enable dynamic graphs for STAN
    )
    
    # Create brain graphs
    graphs = graph_creator.batch_create_graphs(features)
    
    return graphs

def train_baseline_model(features, labels):
    """Train a baseline SVM model on the connectivity matrices"""
    print("\n=== Training Baseline SVM Model ===")
    
    # Extract connectivity matrices as features, ensuring consistent shape
    # First, find the minimum shape to use as reference
    min_shape = float('inf')
    for feature in features:
        shape = feature['connectivity_matrix'].shape[0]
        if shape < min_shape:
            min_shape = shape
    
    print(f"Standardizing all matrices to {min_shape}x{min_shape}")
    
    # Create the feature vectors with standardized shapes
    X = []
    for feature in features:
        # Crop the matrix if needed
        matrix = feature['connectivity_matrix'][:min_shape, :min_shape]
        # Also handle any potential NaN values
        matrix = np.nan_to_num(matrix, nan=0.0)
        X.append(matrix.flatten())
    
    X = np.array(X)
    y = np.array(labels)
    
    # Initialize SVM
    svm = SVC(kernel='linear', probability=True)
    
    # Use 5-fold cross-validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Store metrics
    metrics_list = []
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        svm.fit(X_train, y_train)
        
        # Predict
        y_pred = svm.predict(X_test)
        y_score = svm.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_binary_metrics(y_test, y_pred, y_score)
        metrics_list.append(metrics)
        
        print(f"Fold {fold+1}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
    
    # Calculate average metrics
    from src.utils.metrics import calculate_cross_validation_metrics
    cv_metrics = calculate_cross_validation_metrics(metrics_list)
    
    print("\nBaseline SVM Model Results:")
    print_metrics_summary(cv_metrics)
    
    return cv_metrics

def run_ablation_study(graphs, labels, use_stan=False):
    """Run an ablation study to test GNN without STAN component"""
    print("\n=== Running Ablation Study: GNN without STAN ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get model parameters from the first graph
    if isinstance(graphs[0], dict):
        graph = graphs[0]['static_graph']
    else:
        graph = graphs[0]
    
    num_node_features = graph.x.shape[1]
    
    # Prepare cross-validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Store metrics
    metrics_list = []
    
    # Convert labels to numpy array
    y = np.array(labels)
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(graphs)), y)):
        print(f"\n--- Fold {fold+1} ---")
        
        # Initialize model (GNN only)
        from src.models.gnn import GNNWithPooling
        
        model = GNNWithPooling(
            num_node_features=num_node_features,
            hidden_channels=32,  # Reduced from 64 to avoid potential instability
            output_channels=1,
            num_layers=2,  # Reduced from 3 to simplify
            dropout=0.1,
            layer_type='gcn',
            pool_type='mean'
        ).to(device)
        
        # Define classifier with direct sigmoid
 # This replaces the GNNClassifier part of your run_ablation_study function
        class GNNClassifier(torch.nn.Module):
            def __init__(self, gnn_model):
                super(GNNClassifier, self).__init__()
                self.gnn = gnn_model
                # Add a small MLP to ensure gradients flow properly
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(1, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1)
                )
                
            def forward(self, data):
                # Handle dict input (from dynamic graphs)
                if isinstance(data, dict):
                    data = data['static_graph']
                
                # Get embeddings from GNN
                x = self.gnn(data.x, data.edge_index, 
                        edge_weight=data.edge_attr, batch=data.batch)
                
                # Ensure shape is correct for the MLP
                x = x.view(-1, 1)
                
                # Pass through MLP
                x = self.mlp(x)
                
                # Apply sigmoid for output between 0 and 1
                return torch.sigmoid(x)
                
        model = GNNClassifier(model).to(device)
        
        # Define optimizer and loss function
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    # Training loop
    model.train()
    num_epochs = 20

    print("Training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Train on each sample in the training set
        for idx in train_idx:
            train_graph = graphs[idx]
            train_label = torch.tensor([[float(labels[idx])]], dtype=torch.float).to(device)
            
            # Move graph to device
            if isinstance(train_graph, dict):
                # Make sure all parts of the graph require gradients
                train_graph['static_graph'] = train_graph['static_graph'].to(device)
                
                # Explicitly make edge weights require gradients if they exist
                if hasattr(train_graph['static_graph'], 'edge_attr') and train_graph['static_graph'].edge_attr is not None:
                    if not train_graph['static_graph'].edge_attr.requires_grad:
                        train_graph['static_graph'].edge_attr = train_graph['static_graph'].edge_attr.detach().clone().requires_grad_(True)
            else:
                train_graph = train_graph.to(device)
                # Explicitly make edge weights require gradients if they exist
                if hasattr(train_graph, 'edge_attr') and train_graph.edge_attr is not None:
                    if not train_graph.edge_attr.requires_grad:
                        train_graph.edge_attr = train_graph.edge_attr.detach().clone().requires_grad_(True)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                output = model(train_graph)
                
                # Ensure shapes match
                if output.shape != train_label.shape:
                    output = output.view(train_label.shape)
                
                # Check for NaN values and clip to prevent instability
                if torch.isnan(output).any():
                    print(f"Warning: NaN detected in output. Resetting to 0.5")
                    # Create a new tensor that requires grad
                    output = torch.tensor([[0.5]], device=device, requires_grad=True)
                    
                # Clip values to ensure numerical stability
                output = torch.clamp(output, min=1e-7, max=1-1e-7)
                
                # Compute loss
                loss = criterion(output, train_label)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"Warning: NaN loss detected. Skipping backward pass.")
                    continue
                    
                epoch_loss += loss.item()
                
                # Make sure loss requires gradient
                if not loss.requires_grad:
                    print("Warning: Loss doesn't require gradients. Skipping backward.")
                    continue
                    
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print("Warning: NaN gradients detected. Skipping parameter update.")
                    continue
                    
                optimizer.step()
                
            except RuntimeError as e:
                print(f"Runtime error during training: {e}")
                continue
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_idx) if len(train_idx) > 0 else float('nan')
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Evaluation
        model.eval()
        fold_y_true = []
        fold_y_pred = []
        fold_y_score = []
        
        with torch.no_grad():
            for idx in test_idx:
                test_graph = graphs[idx]
                test_label = float(labels[idx])
                fold_y_true.append(test_label)
                
                # Move graph to device
                if isinstance(test_graph, dict):
                    test_graph['static_graph'] = test_graph['static_graph'].to(device)
                else:
                    test_graph = test_graph.to(device)
                
                # Forward pass
                test_output = model(test_graph)
                if test_output.shape != (1, 1):
                    test_output = test_output.view(1, 1)
                
                # Check for NaN and replace with 0.5
                if torch.isnan(test_output).any():
                    test_output = torch.tensor([[0.5]], device=device)
                
                test_score = test_output.item()
                test_pred = 1 if test_score > 0.5 else 0
                
                fold_y_pred.append(test_pred)
                fold_y_score.append(test_score)
        
        # Calculate metrics for this fold
        metrics = calculate_binary_metrics(
            np.array(fold_y_true), 
            np.array(fold_y_pred), 
            np.array(fold_y_score)
        )
        metrics_list.append(metrics)
        
        # Handle potential None in roc_auc
        roc_auc_value = metrics.get('roc_auc')
        roc_auc_str = f"{roc_auc_value:.4f}" if roc_auc_value is not None else "N/A"
        
        print(f"Fold {fold+1} metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC AUC: {roc_auc_str}")
    
    # Calculate overall metrics
    cv_metrics = calculate_cross_validation_metrics(metrics_list)
    
    print("\nGNN-only Model Results:")
    print_metrics_summary(cv_metrics)
    
    return cv_metrics

def compare_models(baseline_metrics, gnn_stan_metrics, gnn_only_metrics):
    """Compare the performance of different models"""
    print("\n=== Model Comparison ===")
    
    # Prepare data for plotting
    models = ['Baseline SVM', 'GNN-only', 'GNN-STAN']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Extract means
    baseline_means = baseline_metrics['means']
    gnn_only_means = gnn_only_metrics['means']
    gnn_stan_means = gnn_stan_metrics['means']
    
    # Collect data
    data = {}
    for metric in metrics:
        data[metric] = [
            baseline_means[metric],
            gnn_only_means[metric],
            gnn_stan_means[metric]
        ]
    
    # Create comparison table
    print("\nPerformance Comparison:")
    print("-" * 70)
    print(f"{'Metric':<15} {'Baseline SVM':<15} {'GNN-only':<15} {'GNN-STAN':<15}")
    print("-" * 70)
    
    for metric in metrics:
        print(f"{metric.capitalize():<15} {baseline_means[metric]:.4f}        {gnn_only_means[metric]:.4f}        {gnn_stan_means[metric]:.4f}")
    
    print("-" * 70)
    
    # Visualize comparison
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [data[m][0] for m in metrics], width, label='Baseline SVM')
    plt.bar(x, [data[m][1] for m in metrics], width, label='GNN-only')
    plt.bar(x + width, [data[m][2] for m in metrics], width, label='GNN-STAN')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, model_data in enumerate([
        [data[m][0] for m in metrics],
        [data[m][1] for m in metrics],
        [data[m][2] for m in metrics]
    ]):
        for j, v in enumerate(model_data):
            offset = (i - 1) * width
            plt.text(j + offset, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison plot saved to 'results/model_comparison.png'")
    
    return data

def visualize_connectivity_matrices(features):
    """Visualize connectivity matrices for each subject"""
    plt.figure(figsize=(15, 10))
    
    # Determine grid size based on number of subjects
    n_subjects = len(features)
    grid_size = int(np.ceil(np.sqrt(n_subjects)))
    
    for i, feature in enumerate(features):
        plt.subplot(grid_size, grid_size, i+1)
        conn_matrix = feature['connectivity_matrix']
        plt.imshow(conn_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Subject {i+1} - {'ADHD' if feature.get('label', 0) == 1 else 'Control'}")
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('results/connectivity_matrices.png', dpi=300, bbox_inches='tight')
    print("Connectivity matrices plot saved to 'results/connectivity_matrices.png'")

def main():
    """Main function to run the enhanced GNN-STAN model test"""
    print("Starting enhanced GNN-STAN model test for ADHD classification using local Peking dataset...")
    
    # Create output directories
    os.makedirs("preprocessed_data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Set the path to your local Peking_1 dataset
    data_dir = "C:\Peking_1"
    
    # Load local dataset (10 subjects for this test)
    dataset = load_local_peking_data(data_dir, max_subjects=6)
    func_files = dataset['func']
    labels = dataset['labels']
    subject_ids = dataset['subject_ids']
    print(f"Loaded {len(func_files)} subjects with labels: {labels}")

    # Step 1: Preprocessing
    print("\n=== Step 1: Preprocessing ===")
    preprocessed_data = run_preprocessing_pipeline(func_files, subject_ids)
    
    # Get time series data
    time_series_data = [data['time_series'] for data in preprocessed_data]
    
    # Step 2: Feature extraction
    print("\n=== Step 2: Feature Extraction ===")
    features = extract_features(time_series_data)
    
    # Add labels to features
    for i, feature in enumerate(features):
        feature['label'] = labels[i]
    
    # Visualize connectivity matrices
    visualize_connectivity_matrices(features)
    
    # Step 3: Brain graph creation
    print("\n=== Step 3: Brain Graph Creation ===")
    graphs = create_brain_graphs(features)
    
    # Step 4: Model comparison
    print("\n=== Step 4: Model Evaluation ===")
    
    # 4.1: Baseline SVM model
    baseline_metrics = train_baseline_model(features, labels)
    
    # 4.2: Ablation study (GNN without STAN)
    gnn_only_metrics = run_ablation_study(graphs, labels)
    
    # 4.3: Full GNN-STAN model
    model, gnn_stan_metrics = train_and_evaluate_gnn_stan_model(graphs, labels)
    
    # 4.4: Compare models
    comparison_data = compare_models(baseline_metrics, gnn_stan_metrics, gnn_only_metrics)
    
    # Step 5: Resource usage analysis
    print("\n=== Step 5: Resource Usage Analysis ===")
    
    # Get current memory usage
    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Estimate memory requirements for scaling up
    avg_memory_per_subject = current_memory / len(func_files)
    estimated_subjects_capacity = int((psutil.virtual_memory().available / 1024 / 1024) // avg_memory_per_subject)
    
    print(f"Current memory usage: {current_memory:.2f} MB")
    print(f"Estimated memory per subject: {avg_memory_per_subject:.2f} MB")
    print(f"Your system can likely handle approximately {estimated_subjects_capacity} subjects in memory")
    print(f"Estimated memory for 100 subjects: {avg_memory_per_subject * 100:.2f} MB")
    
    # Recommendations
    print("\n=== Recommendations for Full Study ===")
    print("1. Implement batch processing for the full dataset to manage memory usage")
    print("2. Add early stopping to reduce training time and prevent overfitting")
    print("3. Use LOSO cross-validation to test generalization across acquisition sites")
    print("4. Consider hyperparameter tuning for the full model")
    print("5. Include visualization of brain regions with high attention weights")
    
    # Save summary to file
    summary = {
        "num_subjects": len(func_files),
        "memory_per_subject_mb": avg_memory_per_subject,
        "estimated_capacity": estimated_subjects_capacity,
        "baseline_accuracy": baseline_metrics['means']['accuracy'],
        "gnn_only_accuracy": gnn_only_metrics['means']['accuracy'],
        "gnn_stan_accuracy": gnn_stan_metrics['means']['accuracy'],
        "baseline_f1": baseline_metrics['means']['f1_score'],
        "gnn_only_f1": gnn_only_metrics['means']['f1_score'],
        "gnn_stan_f1": gnn_stan_metrics['means']['f1_score'],
    }
    
    with open("results/enhanced_test_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print("\nSummary saved to 'results/enhanced_test_summary.txt'")
    print("\nEnhanced test completed successfully!")

if __name__ == "__main__":
    main()