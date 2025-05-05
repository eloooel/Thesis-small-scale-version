"""
Hybrid GNN-STAN model for ADHD classification.
Combines Graph Neural Networks (GNN) with Spatio-Temporal Attention Networks (STAN).
Added visualization capabilities for inspection of intermediate outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from .gnn import GNN, StableGNNWithPooling
from .stan import STAN, DynamicSTAN

class GNN_STAN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_time_points=None,
                 output_dim=1, gnn_layers=3, gnn_type='gcn', use_edge_weights=True,
                 dropout=0.1, use_dynamic=False, bidirectional=True):
        """
        Initialize the GNN-STAN model.
        
        Parameters:
        -----------
        num_node_features : int
            Number of input features per node
        hidden_channels : int
            Dimension of hidden features
        num_time_points : int, optional
            Number of time points (required if use_dynamic=True)
        output_dim : int, optional
            Dimension of output features (1 for binary classification)
        gnn_layers : int, optional
            Number of GNN layers
        gnn_type : str, optional
            Type of GNN layer to use
        use_edge_weights : bool, optional
            Whether to use edge weights in GNN
        dropout : float, optional
            Dropout probability
        use_dynamic : bool, optional
            Whether to use dynamic version with time series of brain graphs
        bidirectional : bool, optional
            Whether to use bidirectional LSTM in dynamic version
        """
        super(GNN_STAN, self).__init__()
        
        self.use_dynamic = use_dynamic
        self.use_edge_weights = use_edge_weights
        self.hidden_channels = hidden_channels
        
        # GNN component
        self.gnn = GNN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            dropout=dropout,
            layer_type=gnn_type
        )
        
        # STAN component
        if use_dynamic:
            if num_time_points is None:
                raise ValueError("num_time_points must be provided when use_dynamic=True")
            
            self.stan = DynamicSTAN(
                hidden_dim=hidden_channels,
                num_time_points=num_time_points,
                output_dim=hidden_channels,  # Intermediate output
                bidirectional=bidirectional,
                dropout=dropout
            )
        else:
            self.stan = STAN(
                hidden_dim=hidden_channels,
                num_time_points=1,  # Static version
                output_dim=hidden_channels,  # Intermediate output
                dropout=dropout
            )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, output_dim)
    
    def forward(self, data, return_attention=False, debug_prints=False, save_visualization=False, 
                subject_id=None, output_dir="results/visualizations"):
        """
        Forward pass through the GNN-STAN model.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data or dict
            Input data
        return_attention : bool, optional
            Whether to return attention weights
        debug_prints : bool, optional
            Whether to print debug information
        save_visualization : bool, optional
            Whether to save visualization of model outputs
        subject_id : str, optional
            Subject ID for visualization file naming
        output_dir : str, optional
            Directory to save visualizations
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape [batch_size, output_dim]
        dict, optional
            Attention weights if return_attention=True
        """
        # Dictionary to store intermediate outputs for visualization
        intermediates = {}
        
        if self.use_dynamic and isinstance(data, dict):
            # Extract static and dynamic graphs
            static_graph = data['static_graph']
            dynamic_graphs = data['dynamic_graphs']
            
            # Process static graph with GNN
            static_node_features = self.gnn(
                static_graph.x, 
                static_graph.edge_index,
                static_graph.edge_attr if self.use_edge_weights else None,
                static_graph.batch
            )
            
            # Store GNN output for visualization
            intermediates['gnn_output'] = static_node_features
            
            if debug_prints:
                print(f"GNN output shape: {static_node_features.shape}")
                print("GNN output sample values:")
                print(static_node_features[0, :5].detach().cpu().numpy())  # First 5 values
            
            # Process dynamic graphs with GNN
            dynamic_node_features = []
            for graph in dynamic_graphs:
                node_features = self.gnn(
                    graph.x, 
                    graph.edge_index,
                    graph.edge_attr if self.use_edge_weights else None,
                    graph.batch
                )
                dynamic_node_features.append(node_features)
            
            # Process through STAN
            if return_attention:
                stan_out, attention_weights = self.stan(
                    dynamic_node_features,
                    return_attention=True
                )
                
                # Store STAN output for visualization
                intermediates['stan_output'] = stan_out
                intermediates['attention_weights'] = attention_weights
                
                if debug_prints:
                    print(f"STAN output shape: {stan_out.shape}")
                    print("STAN output sample values:")
                    print(stan_out[0, :5].detach().cpu().numpy())  # First 5 values
                    
                    if 'spatial_weights' in attention_weights:
                        print("Spatial attention shape:", attention_weights['spatial_weights'].shape)
                    if 'temporal_weights' in attention_weights:
                        print("Temporal attention shape:", attention_weights['temporal_weights'].shape)
                
                # Final classification
                logits = self.classifier(stan_out)
                intermediates['logits'] = logits
                intermediates['probabilities'] = torch.sigmoid(logits)
                
                if debug_prints:
                    print(f"Logit shape: {logits.shape}")
                    print(f"Logit value: {logits.item():.4f}")
                    print(f"Probability: {torch.sigmoid(logits).item():.4f}")
                
                # Save visualization if requested
                if save_visualization:
                    self._save_visualizations(intermediates, subject_id, output_dir)
                
                return logits, attention_weights
            else:
                stan_out = self.stan(dynamic_node_features)
                intermediates['stan_output'] = stan_out
                
                # Final classification
                logits = self.classifier(stan_out)
                intermediates['logits'] = logits
                intermediates['probabilities'] = torch.sigmoid(logits)
                
                # Save visualization if requested
                if save_visualization:
                    self._save_visualizations(intermediates, subject_id, output_dir)
                
                return logits
        else:
            # Static version
            if isinstance(data, dict):
                data = data['static_graph']
            
            # Process graph with GNN
            node_features = self.gnn(
                data.x, 
                data.edge_index,
                data.edge_attr if self.use_edge_weights else None
            )
            intermediates['gnn_output'] = node_features
            
            if debug_prints:
                print(f"GNN output shape: {node_features.shape}")
                print("GNN output sample values:")
                print(node_features[0, :5].detach().cpu().numpy())  # First 5 values
            
            # Process through STAN
            if return_attention:
                stan_out, attention_weights = self.stan(
                    node_features,
                    batch=data.batch,
                    return_attention=True
                )
                intermediates['stan_output'] = stan_out
                intermediates['attention_weights'] = attention_weights
                
                if debug_prints:
                    print(f"STAN output shape: {stan_out.shape}")
                    print("STAN output sample values:")
                    print(stan_out[0, :5].detach().cpu().numpy())  # First 5 values
                
                # Final classification
                logits = self.classifier(stan_out)
                intermediates['logits'] = logits
                intermediates['probabilities'] = torch.sigmoid(logits)
                
                if debug_prints:
                    print(f"Logit shape: {logits.shape}")
                    print(f"Logit value: {logits.item():.4f}")
                    print(f"Probability: {torch.sigmoid(logits).item():.4f}")
                
                # Save visualization if requested
                if save_visualization:
                    self._save_visualizations(intermediates, subject_id, output_dir)
                
                return logits, attention_weights
            else:
                stan_out = self.stan(node_features, batch=data.batch)
                intermediates['stan_output'] = stan_out
                
                # Final classification
                logits = self.classifier(stan_out)
                intermediates['logits'] = logits
                intermediates['probabilities'] = torch.sigmoid(logits)
                
                # Save visualization if requested
                if save_visualization:
                    self._save_visualizations(intermediates, subject_id, output_dir)
                
                return logits
    
    def _save_visualizations(self, intermediates, subject_id=None, output_dir="results/visualizations"):
        """
        Save visualizations of model outputs.
        
        Parameters:
        -----------
        intermediates : dict
            Dictionary containing intermediate outputs
        subject_id : str, optional
            Subject ID for file naming
        output_dir : str, optional
            Directory to save visualizations
        """
        # Create output directory
        if subject_id is not None:
            output_dir = os.path.join(output_dir, f"subject_{subject_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Visualize GNN output
        if 'gnn_output' in intermediates:
            gnn_np = intermediates['gnn_output'].detach().cpu().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.bar(range(min(self.hidden_channels, gnn_np.shape[1])), 
                   gnn_np[0, :min(self.hidden_channels, gnn_np.shape[1])])
            plt.xlabel('Embedding Dimension')
            plt.ylabel('Value')
            plt.title('GNN Embedding Visualization')
            plt.savefig(os.path.join(output_dir, 'gnn_output.png'))
            plt.close()
            
            # Save raw data
            np.save(os.path.join(output_dir, 'gnn_output.npy'), gnn_np)
        
        # 2. Visualize STAN output
        if 'stan_output' in intermediates:
            stan_np = intermediates['stan_output'].detach().cpu().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.bar(range(min(self.hidden_channels, stan_np.shape[1])), 
                   stan_np[0, :min(self.hidden_channels, stan_np.shape[1])])
            plt.xlabel('Embedding Dimension')
            plt.ylabel('Value')
            plt.title('STAN Embedding Visualization')
            plt.savefig(os.path.join(output_dir, 'stan_output.png'))
            plt.close()
            
            # Save raw data
            np.save(os.path.join(output_dir, 'stan_output.npy'), stan_np)
        
        # 3. Visualize attention weights (if available)
        if 'attention_weights' in intermediates:
            attention_weights = intermediates['attention_weights']
            
            if 'spatial_weights' in attention_weights:
                spatial_np = attention_weights['spatial_weights'].detach().cpu().numpy()
                
                # Check if it's 1D or 2D
                if spatial_np.ndim == 1:
                    plt.figure(figsize=(10, 4))
                    plt.bar(range(len(spatial_np)), spatial_np)
                    plt.xlabel('Brain Region')
                    plt.ylabel('Attention Weight')
                    plt.title('Spatial Attention Weights')
                else:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(spatial_np, aspect='auto')
                    plt.colorbar()
                    plt.xlabel('Brain Region')
                    plt.ylabel('Batch/Sample')
                    plt.title('Spatial Attention Weights')
                
                plt.savefig(os.path.join(output_dir, 'spatial_attention.png'))
                plt.close()
                
                # Save raw data
                np.save(os.path.join(output_dir, 'spatial_attention.npy'), spatial_np)
            
            if 'temporal_weights' in attention_weights:
                temporal_np = attention_weights['temporal_weights'].detach().cpu().numpy()
                
                # Check if it's 1D or 2D
                if temporal_np.ndim == 1:
                    plt.figure(figsize=(10, 4))
                    plt.bar(range(len(temporal_np)), temporal_np)
                    plt.xlabel('Time Point')
                    plt.ylabel('Attention Weight')
                    plt.title('Temporal Attention Weights')
                else:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(temporal_np, aspect='auto')
                    plt.colorbar()
                    plt.xlabel('Time Point')
                    plt.ylabel('Batch/Sample')
                    plt.title('Temporal Attention Weights')
                
                plt.savefig(os.path.join(output_dir, 'temporal_attention.png'))
                plt.close()
                
                # Save raw data
                np.save(os.path.join(output_dir, 'temporal_attention.npy'), temporal_np)
        
        # 4. Save prediction values
        if 'logits' in intermediates and 'probabilities' in intermediates:
            logits = intermediates['logits'].detach().cpu().numpy()
            probs = intermediates['probabilities'].detach().cpu().numpy()
            
            with open(os.path.join(output_dir, 'prediction_summary.txt'), 'w') as f:
                f.write(f"Logit value: {logits.item():.4f}\n")
                f.write(f"Probability: {probs.item():.4f}\n")
                f.write(f"Prediction: {'ADHD' if probs.item() > 0.5 else 'Control'}\n")
                
                if 'gnn_output' in intermediates:
                    f.write(f"\nGNN output shape: {intermediates['gnn_output'].shape}\n")
                
                if 'stan_output' in intermediates:
                    f.write(f"STAN output shape: {intermediates['stan_output'].shape}\n")
                
                if 'attention_weights' in intermediates:
                    attention = intermediates['attention_weights']
                    if 'spatial_weights' in attention:
                        f.write(f"Spatial attention shape: {attention['spatial_weights'].shape}\n")
                    if 'temporal_weights' in attention:
                        f.write(f"Temporal attention shape: {attention['temporal_weights'].shape}\n")

class GNN_STAN_Classifier(nn.Module):
    """
    Complete GNN-STAN classifier with sigmoid activation for binary classification.
    """
    
    def __init__(self, num_node_features, hidden_channels, num_time_points=None,
                 gnn_layers=3, gnn_type='gcn', use_edge_weights=True,
                 dropout=0.1, use_dynamic=False, bidirectional=True):
        """
        Initialize the GNN-STAN classifier.
        
        Parameters:
        -----------
        num_node_features : int
            Number of input features per node
        hidden_channels : int
            Dimension of hidden features
        num_time_points : int, optional
            Number of time points (required if use_dynamic=True)
        gnn_layers : int, optional
            Number of GNN layers
        gnn_type : str, optional
            Type of GNN layer to use
        use_edge_weights : bool, optional
            Whether to use edge weights in GNN
        dropout : float, optional
            Dropout probability
        use_dynamic : bool, optional
            Whether to use dynamic version with time series of brain graphs
        bidirectional : bool, optional
            Whether to use bidirectional LSTM in dynamic version
        """
        super(GNN_STAN_Classifier, self).__init__()
        
        # GNN-STAN model
        self.model = GNN_STAN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_time_points=num_time_points,
            output_dim=1,  # Binary classification
            gnn_layers=gnn_layers,
            gnn_type=gnn_type,
            use_edge_weights=use_edge_weights,
            dropout=dropout,
            use_dynamic=use_dynamic,
            bidirectional=bidirectional
        )
    
    def forward(self, data, return_attention=False, return_logits=False, 
                debug_prints=False, save_visualization=False, subject_id=None):
        """
        Forward pass through the GNN-STAN classifier.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data or dict
            Input data
        return_attention : bool, optional
            Whether to return attention weights
        return_logits : bool, optional
            Whether to return logits instead of probabilities
        debug_prints : bool, optional
            Whether to print debug information
        save_visualization : bool, optional
            Whether to save visualization of model outputs
        subject_id : str, optional
            Subject ID for visualization file naming
            
        Returns:
        --------
        torch.Tensor
            Output probabilities or logits of shape [batch_size, 1]
        dict, optional
            Attention weights if return_attention=True
        """
        if return_attention:
            logits, attention_weights = self.model(
                data, 
                return_attention=True, 
                debug_prints=debug_prints,
                save_visualization=save_visualization,
                subject_id=subject_id
            )
            
            if return_logits:
                return logits, attention_weights
            else:
                probs = torch.sigmoid(logits)
                return probs, attention_weights
        else:
            logits = self.model(
                data, 
                debug_prints=debug_prints,
                save_visualization=save_visualization,
                subject_id=subject_id
            )
            
            if return_logits:
                return logits
            else:
                probs = torch.sigmoid(logits)
                return probs
    
    def predict(self, data, threshold=0.5, save_visualization=False, subject_id=None):
        """
        Make binary predictions.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data or dict
            Input data
        threshold : float, optional
            Classification threshold
        save_visualization : bool, optional
            Whether to save visualization of model outputs
        subject_id : str, optional
            Subject ID for visualization file naming
            
        Returns:
        --------
        torch.Tensor
            Binary predictions of shape [batch_size, 1]
        """
        probs = self.forward(
            data, 
            save_visualization=save_visualization,
            subject_id=subject_id
        )
        return (probs >= threshold).float()
    
    def visualize_prediction(self, data, subject_id=None, output_dir="results/visualizations"):
        """
        Visualize model's prediction process for a given input.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data or dict
            Input data
        subject_id : str, optional
            Subject ID for visualization file naming
        output_dir : str, optional
            Directory to save visualizations
        
        Returns:
        --------
        float
            Prediction probability
        """
        # Forward pass with visualization enabled
        if isinstance(subject_id, int):
            subject_id = str(subject_id)
            
        probs, attention_weights = self.forward(
            data,
            return_attention=True,
            debug_prints=True,
            save_visualization=True,
            subject_id=subject_id
        )
        
        # Print prediction
        prob_value = probs.item()
        prediction = "ADHD" if prob_value > 0.5 else "Control"
        print(f"\nPrediction: {prediction} (Probability: {prob_value:.4f})")
        print(f"Visualizations saved to: {os.path.join(output_dir, f'subject_{subject_id}')}")
        
        return prob_value