"""
Small-scale test of the GNN-STAN model for ADHD classification using local Peking_1 dataset.

This script uses a local dataset (2 subjects from Peking_1) and runs a simple test
to verify the model works and to estimate resource requirements.
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

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.preprocessing.preprocessing import Preprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.brain_graph import BrainGraphCreator
from src.models.gnn_stan import GNN_STAN_Classifier

def load_local_peking_data(data_dir):
    """
    Load the local Peking_1 dataset (2 subjects)
    
    Parameters:
    -----------
    data_dir : str
        Path to the Peking_1 dataset directory
    
    Returns:
    --------
    dict
        Dictionary containing functional files, subject IDs, and labels
    """
    print(f"Loading local Peking_1 dataset from {data_dir}")
    
    # Find all functional MRI files
    func_pattern = os.path.join(data_dir, "sub-*", "ses-*", "func", "*_task-rest_*.nii.gz")
    func_files = sorted(glob.glob(func_pattern))
    
    if not func_files:
        raise FileNotFoundError(f"No functional MRI files found in {data_dir}")
    
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
    
    print(f"Found {len(func_files)} subjects:")
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

@measure_resource_usage
def train_and_evaluate_model(graphs, labels):
    """Train and evaluate the GNN-STAN model"""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get model parameters from the first graph
    if isinstance(graphs[0], dict):
        graph = graphs[0]['static_graph']
        num_time_points = len(graphs[0]['dynamic_graphs'])
    else:
        graph = graphs[0]
        num_time_points = 10  # Default
    
    num_node_features = graph.x.shape[1]
    
    # Initialize model
    model = GNN_STAN_Classifier(
        num_node_features=num_node_features,
        hidden_channels=64,
        num_time_points=num_time_points,
        gnn_layers=3,
        gnn_type='gcn',
        use_edge_weights=True,
        dropout=0.1,
        use_dynamic=True,
        bidirectional=True
    ).to(device)
    
    # Print model summary
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train on one subject, test on the other
    train_graph = graphs[0]
    train_label = torch.tensor([float(labels[0])], dtype=torch.float).to(device)
    
    test_graph = graphs[1]
    test_label = torch.tensor([float(labels[1])], dtype=torch.float).to(device)
    
    # Move graphs to device
    if isinstance(train_graph, dict):
        train_graph['static_graph'] = train_graph['static_graph'].to(device)
        train_graph['dynamic_graphs'] = [g.to(device) for g in train_graph['dynamic_graphs']]
    else:
        train_graph = train_graph.to(device)
    
    if isinstance(test_graph, dict):
        test_graph['static_graph'] = test_graph['static_graph'].to(device)
        test_graph['dynamic_graphs'] = [g.to(device) for g in test_graph['dynamic_graphs']]
    else:
        test_graph = test_graph.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    
    # Training loop
    model.train()
    num_epochs = 20
    losses = []
    
    print("\nTraining model...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_graph)
        
        # Compute loss
        loss = criterion(output, train_label)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('results/training_loss.png')
    print("Training loss plot saved to 'results/training_loss.png'")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Forward pass
        test_output = model(test_graph)
        test_pred = (test_output > 0.5).float()
        
        # Compute accuracy
        test_acc = (test_pred == test_label).float().mean().item()
        
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"True label: {labels[1]}, Predicted: {test_pred.item():.0f}")
    
    # Extract attention weights for visualization
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(test_graph, return_attention=True)
        
        # Save attention weights for analysis
        if isinstance(attention_weights, dict):
            if 'spatial_weights' in attention_weights:
                spatial_weights = attention_weights['spatial_weights']
                
                # Plot spatial attention weights if possible
                if isinstance(spatial_weights, torch.Tensor) and spatial_weights.dim() == 2:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(spatial_weights.cpu().numpy(), cmap='hot', aspect='auto')
                    plt.colorbar()
                    plt.title('Spatial Attention Weights')
                    plt.savefig('results/spatial_attention.png')
                    print("Spatial attention weights plot saved to 'results/spatial_attention.png'")
            
            if 'temporal_weights' in attention_weights:
                temporal_weights = attention_weights['temporal_weights']
                
                # Plot temporal attention weights if possible
                if isinstance(temporal_weights, torch.Tensor):
                    plt.figure(figsize=(12, 6))
                    plt.plot(temporal_weights.squeeze().cpu().numpy(), marker='o')
                    plt.title('Temporal Attention Weights')
                    plt.xlabel('Time Window')
                    plt.ylabel('Attention Weight')
                    plt.grid(True)
                    plt.savefig('results/temporal_attention.png')
                    print("Temporal attention weights plot saved to 'results/temporal_attention.png'")
    
    return model, losses, test_acc

def visualize_connectivity_matrices(features):
    """Visualize connectivity matrices for each subject"""
    plt.figure(figsize=(15, 6))
    
    for i, feature in enumerate(features):
        plt.subplot(1, len(features), i+1)
        conn_matrix = feature['connectivity_matrix']
        plt.imshow(conn_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Subject {i+1} - {'ADHD' if feature.get('label', 0) == 1 else 'Control'}")
    
    plt.tight_layout()
    plt.savefig('results/connectivity_matrices.png')
    print("Connectivity matrices plot saved to 'results/connectivity_matrices.png'")

def main():
    """Main function to run the small-scale test"""
    print("Starting small-scale GNN-STAN model test for ADHD classification using local Peking dataset...")
    
    # Create output directories
    os.makedirs("preprocessed_data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Set the path to your local Peking_1 dataset
    data_dir = "C:/Users/Raphael/OneDrive - ust.edu.ph/Documents/SY 2024-2025/2ND SEM/Thesis/dataset/Peking_1"
    
    # Load local dataset
    dataset = load_local_peking_data(data_dir)
    func_files = dataset['func']
    labels = dataset['labels']
    subject_ids = dataset['subject_ids']
    
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
    
    # Step 4: Model training and evaluation
    print("\n=== Step 4: Model Training and Evaluation ===")
    model, losses, test_acc = train_and_evaluate_model(graphs, labels)
    
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
    print("\n=== Recommendations ===")
    print("1. Consider using batch processing for the full dataset")
    print("2. Implement early stopping to reduce training time")
    print("3. If available, use a GPU to significantly speed up model training")
    print("4. For cross-validation with the full dataset, use proper LOSO and stratified k-fold strategies")
    
    # Save summary to file
    summary = {
        "num_subjects": len(func_files),
        "labels": labels,
        "test_accuracy": test_acc,
        "final_loss": losses[-1],
        "memory_per_subject_mb": avg_memory_per_subject,
        "estimated_capacity": estimated_subjects_capacity
    }
    
    with open("results/mini_test_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print("\nSummary saved to 'results/mini_test_summary.txt'")
    print("\nSmall-scale test completed successfully!")

if __name__ == "__main__":
    main()