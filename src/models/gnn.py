import os
import numpy as np
import nibabel as nib
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from nilearn import masking, image, connectome
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time
import psutil

class GNN(torch.nn.Module):
    """Graph Neural Network component of the hybrid model"""
    def __init__(self, num_node_features, hidden_channels):
        super(GNN, self).__init__()
        # Graph convolutional layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        # First Graph Convolution
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Second Graph Convolution
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Third Graph Convolution
        x = self.conv3(x, edge_index, edge_weight)
        
        return x

class STAN(torch.nn.Module):
    """Spatio-Temporal Attention Network component"""
    def __init__(self, hidden_channels, num_time_points):
        super(STAN, self).__init__()
        # Spatial attention
        self.spatial_attention = torch.nn.Linear(hidden_channels, 1)
        
        # Temporal attention
        self.temporal_attention = torch.nn.Linear(num_time_points, num_time_points)
        
    def forward(self, x, batch=None):
        # Apply spatial attention
        spatial_weights = F.softmax(self.spatial_attention(x), dim=0)
        x_weighted = x * spatial_weights
        
        # If we have batches, apply global pooling per graph
        if batch is not None:
            x_pooled = global_mean_pool(x_weighted, batch)
        else:
            x_pooled = torch.mean(x_weighted, dim=0)
        
        # Reshape for temporal attention (assuming time is the last dimension)
        x_pooled = x_pooled.unsqueeze(0) if x_pooled.dim() == 1 else x_pooled
        
        # Apply temporal attention
        temporal_weights = F.softmax(self.temporal_attention(x_pooled), dim=1)
        output = torch.matmul(temporal_weights, x_pooled)
        
        return output

class GNN_STAN(torch.nn.Module):
    """Hybrid GNN-STAN model for ADHD classification"""
    def __init__(self, num_node_features, hidden_channels, num_time_points):
        super(GNN_STAN, self).__init__()
        self.gnn = GNN(num_node_features, hidden_channels)
        self.stan = STAN(hidden_channels, num_time_points)
        
        # Final classification layer
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Apply GNN
        x = self.gnn(x, edge_index, edge_weight)
        
        # Apply STAN
        x = self.stan(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return torch.sigmoid(x)

def preprocess_fmri(fmri_path, output_dir):
    """
    Simplified preprocessing for rs-fMRI data
    1. Motion correction
    2. Spatial normalization 
    3. Temporal filtering
    4. Denoising
    """
    print(f"Preprocessing {fmri_path}")
    # Load the fMRI data
    img = nib.load(fmri_path)
    
    # Basic motion correction and denoising using nilearn
    cleaned_img = image.clean_img(
        img,
        detrend=True,
        standardize='zscore',
        low_pass=0.08,
        high_pass=0.009,
        t_r=2.0  # Adjust based on your data's TR
    )
    
    # Create a brain mask
    mask_img = masking.compute_epi_mask(cleaned_img)
    
    # Apply mask to get time series data
    masked_data = masking.apply_mask(cleaned_img, mask_img)
    
    # Save preprocessed data
    subject_id = os.path.basename(os.path.dirname(fmri_path))
    output_file = os.path.join(output_dir, f"{subject_id}_preprocessed.npy")
    np.save(output_file, masked_data)
    
    # Also save the mask for later use
    mask_file = os.path.join(output_dir, f"{subject_id}_mask.nii.gz")
    nib.save(mask_img, mask_file)
    
    return output_file, mask_file

def extract_connectivity_features(time_series_file, atlas_labels=None):
    """
    Extract functional connectivity features from time series data
    """
    # Load time series data
    time_series = np.load(time_series_file)
    
    # If no atlas labels provided, create simpler ROIs by averaging neighboring voxels
    # This is a major simplification for the mini-model
    if atlas_labels is None:
        # Reduce dimensionality by averaging neighboring voxels
        # Assuming time_series shape is (timepoints, voxels)
        n_voxels = time_series.shape[1]
        n_rois = min(200, n_voxels // 10)  # Create ~200 ROIs or fewer if not enough voxels
        
        # Simple approach: average groups of voxels
        voxels_per_roi = n_voxels // n_rois
        roi_time_series = np.zeros((time_series.shape[0], n_rois))
        
        for i in range(n_rois):
            start_idx = i * voxels_per_roi
            end_idx = start_idx + voxels_per_roi if i < n_rois - 1 else n_voxels
            roi_time_series[:, i] = np.mean(time_series[:, start_idx:end_idx], axis=1)
    else:
        # Use provided atlas labels to aggregate voxels
        # This would be implemented for the full model
        roi_time_series = time_series
    
    # Compute connectivity matrix
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([roi_time_series])[0]
    
    # Replace potential NaNs with zeros
    correlation_matrix = np.nan_to_fill(correlation_matrix, 0)
    
    return correlation_matrix, roi_time_series

def create_graph_from_connectivity(correlation_matrix, roi_time_series, threshold=0.5):
    """
    Create a PyTorch Geometric graph from connectivity matrix
    """
    # Apply threshold to create binary adjacency matrix
    adj_matrix = (correlation_matrix > threshold).astype(np.float32)
    
    # Get edges (connections between ROIs)
    edges = np.where(adj_matrix > 0)
    edge_index = torch.tensor(np.array([edges[0], edges[1]]), dtype=torch.long)
    
    # Edge weights are the correlation values
    edge_weights = torch.tensor(correlation_matrix[edges], dtype=torch.float)
    
    # Node features are the time series of each ROI
    # For simplicity, we'll use the mean and std of each ROI's time series as features
    n_rois = roi_time_series.shape[1]
    node_features = torch.zeros((n_rois, 2), dtype=torch.float)
    
    for i in range(n_rois):
        node_features[i, 0] = torch.tensor(np.mean(roi_time_series[:, i]))
        node_features[i, 1] = torch.tensor(np.std(roi_time_series[:, i]))
    
    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)
    
    return data

def train_eval_model(model, data_list, labels, n_epochs=10):
    """
    Train and evaluate the model using basic cross-validation
    """
    # Simple train/test split for the mini model
    train_data = [data_list[0]]
    train_label = [labels[0]]
    test_data = [data_list[1]]
    test_label = [labels[1]]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    # Training loop
    model.train()
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    training_losses = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = torch.tensor([[train_label[0]]], dtype=torch.float)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = (out > 0.5).float().item()
            predictions.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score([test_label[0]], predictions)
    
    # Performance statistics
    training_time = time.time() - start_time
    memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs+1), training_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    return accuracy, training_time, memory_used

def main():
    """
    Main function to run the small-scale GNN-STAN model
    """
    # Create output directory for preprocessed data
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to subject data (adjust these to your actual paths)
    subject_paths = [
        "/path/to/subject1/rest.nii.gz",
        "/path/to/subject2/rest.nii.gz"
    ]
    
    # Labels (1 for ADHD, 0 for control)
    # For this mini-model, we'll assume one of each for simple train/test
    labels = [1, 0]  # Adjust based on your actual subjects
    
    # Preprocess data
    preprocessed_files = []
    mask_files = []
    
    for path in subject_paths:
        if os.path.exists(path):
            pre_file, mask_file = preprocess_fmri(path, output_dir)
            preprocessed_files.append(pre_file)
            mask_files.append(mask_file)
        else:
            print(f"Warning: File {path} does not exist. Please check the path.")
    
    # Extract features and create graphs
    graph_data_list = []
    
    for pre_file in preprocessed_files:
        # Extract connectivity features
        conn_matrix, roi_time_series = extract_connectivity_features(pre_file)
        
        # Create graph
        graph_data = create_graph_from_connectivity(conn_matrix, roi_time_series)
        graph_data_list.append(graph_data)
    
    # Initialize the model
    num_node_features = graph_data_list[0].x.shape[1]
    hidden_channels = 64
    num_time_points = 10  # Simplified for the mini-model
    
    model = GNN_STAN(num_node_features, hidden_channels, num_time_points)
    
    # Train and evaluate
    accuracy, training_time, memory_used = train_eval_model(
        model, graph_data_list, labels, n_epochs=10
    )
    
    # Save results
    results = {
        "accuracy": accuracy,
        "training_time": training_time,
        "memory_used": memory_used
    }
    
    with open("mini_model_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print("\nResults saved to mini_model_results.txt")
    print("\nSmall-scale GNN-STAN model test completed!")

if __name__ == "__main__":
    main()