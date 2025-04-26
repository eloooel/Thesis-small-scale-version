"""
Spatio-Temporal Attention Network (STAN) module for ADHD classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important brain regions.
    """
    
    def __init__(self, hidden_dim, attention_dim=None):
        """
        Initialize the spatial attention module.
        
        Parameters:
        -----------
        hidden_dim : int
            Dimension of hidden features
        attention_dim : int, optional
            Dimension of attention space
        """
        super(SpatialAttention, self).__init__()
        
        if attention_dim is None:
            attention_dim = hidden_dim // 2
        
        # MLP for computing attention scores
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the spatial attention module.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features of shape [num_nodes, hidden_dim]
            
        Returns:
        --------
        torch.Tensor
            Weighted node features of shape [num_nodes, hidden_dim]
        torch.Tensor
            Attention weights of shape [num_nodes, 1]
        """
        # Compute attention scores
        attention_scores = self.attention_mlp(x)  # Shape: [num_nodes, 1]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=0)  # Shape: [num_nodes, 1]
        
        # Apply attention weights to node features
        weighted_features = x * attention_weights  # Shape: [num_nodes, hidden_dim]
        
        return weighted_features, attention_weights

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time points.
    """
    
    def __init__(self, hidden_dim, num_time_points, attention_dim=None):
        """
        Initialize the temporal attention module.
        
        Parameters:
        -----------
        hidden_dim : int
            Dimension of hidden features
        num_time_points : int
            Number of time points
        attention_dim : int, optional
            Dimension of attention space
        """
        super(TemporalAttention, self).__init__()
        
        if attention_dim is None:
            attention_dim = hidden_dim // 2
        
        # MLP for computing attention scores
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, num_time_points)
        )
    
    def forward(self, x):
        """
        Forward pass through the temporal attention module.
        
        Parameters:
        -----------
        x : torch.Tensor
            Temporal features of shape [batch_size, num_time_points, hidden_dim]
            
        Returns:
        --------
        torch.Tensor
            Weighted temporal features of shape [batch_size, hidden_dim]
        torch.Tensor
            Attention weights of shape [batch_size, num_time_points]
        """
        # Get batch size
        batch_size = x.size(0)
        
        # Compute attention scores for each sample in batch
        attention_scores = []
        for i in range(batch_size):
            # For each sample, get the mean feature across time points
            mean_feature = torch.mean(x[i], dim=0, keepdim=True)  # Shape: [1, hidden_dim]
            score = self.attention_mlp(mean_feature)  # Shape: [1, num_time_points]
            attention_scores.append(score)
        
        # Stack attention scores
        attention_scores = torch.cat(attention_scores, dim=0)  # Shape: [batch_size, num_time_points]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, num_time_points]
        
        # Apply attention weights to temporal features
        weighted_sum = torch.bmm(
            attention_weights.unsqueeze(1),  # Shape: [batch_size, 1, num_time_points]
            x  # Shape: [batch_size, num_time_points, hidden_dim]
        ).squeeze(1)  # Shape: [batch_size, hidden_dim]
        
        return weighted_sum, attention_weights

class STAN(nn.Module):
    """
    Spatio-Temporal Attention Network for ADHD classification.
    
    This module implements a two-level attention mechanism:
    1. Spatial attention to focus on important brain regions
    2. Temporal attention to focus on important time points
    """
    
    def __init__(self, hidden_dim, num_time_points, output_dim=1,
                 spatial_attention_dim=None, temporal_attention_dim=None,
                 dropout=0.1):
        """
        Initialize the STAN module.
        
        Parameters:
        -----------
        hidden_dim : int
            Dimension of hidden features
        num_time_points : int
            Number of time points
        output_dim : int, optional
            Dimension of output features
        spatial_attention_dim : int, optional
            Dimension of spatial attention space
        temporal_attention_dim : int, optional
            Dimension of temporal attention space
        dropout : float, optional
            Dropout probability
        """
        super(STAN, self).__init__()
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(
            hidden_dim, 
            attention_dim=spatial_attention_dim
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim, 
            num_time_points, 
            attention_dim=temporal_attention_dim
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.hidden_dim = hidden_dim
        self.num_time_points = num_time_points
    
    def forward(self, x, batch=None, return_attention=False):
        """
        Forward pass through the STAN module.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features
            If dynamic: shape [batch_size, num_time_points, num_nodes, hidden_dim]
            If static: shape [num_nodes, hidden_dim]
        batch : torch.Tensor, optional
            Batch vector of shape [num_nodes] to identify node batch assignments
            Only used if x is static
        return_attention : bool, optional
            Whether to return attention weights
            
        Returns:
        --------
        torch.Tensor
            Output features of shape [batch_size, output_dim]
        dict, optional
            Attention weights if return_attention=True
        """
        # Check if input is dynamic or static
        is_dynamic = len(x.shape) == 4
        
        if is_dynamic:
            # Dynamic input: [batch_size, num_time_points, num_nodes, hidden_dim]
            batch_size, num_time_points, num_nodes, hidden_dim = x.shape
            
            # Apply spatial attention for each time point
            spatial_outputs = []
            spatial_attention_weights = []
            
            for t in range(num_time_points):
                spatial_output, spatial_weights = self.spatial_attention(x[:, t])  # For each time point
                spatial_outputs.append(spatial_output)
                spatial_attention_weights.append(spatial_weights)
            
            # Stack spatial outputs
            spatial_outputs = torch.stack(spatial_outputs, dim=1)  # Shape: [batch_size, num_time_points, hidden_dim]
            
            # Apply temporal attention
            temporal_output, temporal_weights = self.temporal_attention(spatial_outputs)
            
            # Apply classifier
            output = self.classifier(temporal_output)
            
            if return_attention:
                return output, {
                    'spatial_weights': spatial_attention_weights,
                    'temporal_weights': temporal_weights
                }
            else:
                return output
        else:
            # Static input: [num_nodes, hidden_dim]
            
            # Apply spatial attention
            spatial_output, spatial_weights = self.spatial_attention(x)  # Shape: [num_nodes, hidden_dim]
            
            # If batch is provided, pool nodes according to batch
            if batch is not None:
                # Pool nodes to get graph-level representations
                from torch_geometric.nn import global_mean_pool
                pooled_output = global_mean_pool(spatial_output, batch)  # Shape: [batch_size, hidden_dim]
            else:
                # If no batch info, assume a single graph
                pooled_output = torch.mean(spatial_output, dim=0, keepdim=True)  # Shape: [1, hidden_dim]
            
            # Apply classifier
            output = self.classifier(pooled_output)
            
            if return_attention:
                return output, {
                    'spatial_weights': spatial_weights
                }
            else:
                return output

class DynamicSTAN(nn.Module):
    """
    Dynamic version of STAN for handling time series of brain graphs.
    
    This module processes a sequence of brain graphs and applies
    spatio-temporal attention to identify important regions and time points.
    """
    
    def __init__(self, hidden_dim, num_time_points, output_dim=1,
                 lstm_layers=1, bidirectional=True, dropout=0.1):
        """
        Initialize the Dynamic STAN module.
        
        Parameters:
        -----------
        hidden_dim : int
            Dimension of hidden features
        num_time_points : int
            Number of time points
        output_dim : int, optional
            Dimension of output features
        lstm_layers : int, optional
            Number of LSTM layers
        bidirectional : bool, optional
            Whether to use bidirectional LSTM
        dropout : float, optional
            Dropout probability
        """
        super(DynamicSTAN, self).__init__()
        
        # LSTM for processing temporal information
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Adjust hidden dimension if bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # STAN module
        self.stan = STAN(
            hidden_dim=lstm_output_dim,
            num_time_points=num_time_points,
            output_dim=output_dim,
            dropout=dropout
        )
    
    def forward(self, x, batch=None, return_attention=False):
        """
        Forward pass through the Dynamic STAN module.
        
        Parameters:
        -----------
        x : torch.Tensor or list
            Input features
            If tensor: shape [batch_size, num_time_points, hidden_dim]
            If list: list of tensors, each of shape [num_nodes, hidden_dim]
        batch : torch.Tensor or list, optional
            Batch vector or list of batch vectors
        return_attention : bool, optional
            Whether to return attention weights
            
        Returns:
        --------
        torch.Tensor
            Output features of shape [batch_size, output_dim]
        dict, optional
            Attention weights if return_attention=True
        """
        # Handle list input (e.g., from dynamic brain graphs)
        if isinstance(x, list):
            # Convert list of graph node features to tensor
            x = torch.stack([torch.mean(g_x, dim=0) for g_x in x], dim=0)  # Shape: [num_time_points, hidden_dim]
            x = x.unsqueeze(0)  # Shape: [1, num_time_points, hidden_dim]
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, num_time_points, lstm_output_dim]
        
        # Process through STAN
        if return_attention:
            output, attention_weights = self.stan(lstm_out, return_attention=True)
            return output, attention_weights
        else:
            output = self.stan(lstm_out)
            return output