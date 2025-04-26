"""
Small-scale test of the GNN-STAN model for ADHD classification.

This script downloads a small sample of the ADHD-200 dataset (2 subjects)
and runs a simple test to verify the model works and to estimate resource requirements.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import psutil
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.preprocessing.preprocessing import Preprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.brain_graph import BrainGraphCreator
from src.models.gnn_stan import GNN_STAN_Classifier

def download_adhd_sample_data():
    """
    Download a small sample from ADHD-200 dataset (2 subjects only)
    """
    print("Downloading sample ADHD-200 data (this might take a few minutes)...")
    from nilearn import datasets
    adhd_dataset = datasets.fetch_adhd(n_subjects=2)
    
    # Print dataset information
    print(f"Dataset has {len(adhd_dataset.func)} functional images")
    print(f"Labels: {adhd_dataset.phenotypic['adhd']}")  # 1 for ADHD, 0 for control
    
    return adhd_dataset

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
    else:
        graph = graphs[0]
    
    num_node_features = graph.x.shape[1]
    
    # Initialize model
    model = GNN_STAN_Classifier(
        num_node_features=num_node_features,
        hidden_channels=64,
        num_time_points=10,  # Adjust based on your dynamic connectivity
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
    num_epochs = 10
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
    plt.savefig('training_loss.png')
    print("Training loss plot saved to 'training_loss.png'")
    
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
                    plt.savefig('spatial_attention.png')
                    print("Spatial attention weights plot saved to 'spatial_attention.png'")
    
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
    plt.savefig('connectivity_matrices.png')
    print("Connectivity matrices plot saved to 'connectivity_matrices.png'")

def main():
    """Main function to run the small-scale test"""
    print("Starting small-scale GNN-STAN model test for ADHD classification...")
    
    # Create output directories
    os.makedirs("preprocessed_data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Download sample data
    dataset = download_adhd_sample_data()
    func_files = dataset.func
    labels = dataset.phenotypic['adhd'].tolist()
    subject_ids = [f"subject_{i+1}" for i in range(len(func_files))]
    
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