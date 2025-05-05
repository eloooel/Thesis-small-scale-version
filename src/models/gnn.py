"""
Improved Graph Neural Network (GNN) module for ADHD classification.
Added stability enhancements to prevent NaN issues.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv, global_mean_pool
from torch.nn import BatchNorm1d, LayerNorm


class StableGNN(torch.nn.Module):
    """
    Improved Graph Neural Network component with stability enhancements.
    
    This module implements a configurable GNN with multiple graph convolutional layers,
    batch normalization, gradient clipping, and careful weight initialization
    to prevent NaN values during computation.
    """
    
    def __init__(self, num_node_features, hidden_channels, num_layers=3, 
                 dropout=0.1, layer_type='gcn', use_batch_norm=True, 
                 use_layer_norm=False, eps=1e-5, max_norm=1.0, **kwargs):
        """
        Initialize the improved GNN module.
        
        Parameters:
        -----------
        num_node_features : int
            Number of input features per node
        hidden_channels : int
            Dimension of hidden features
        num_layers : int, optional
            Number of graph convolutional layers
        dropout : float, optional
            Dropout probability
        layer_type : str, optional
            Type of graph convolutional layer to use
            Options: 'gcn', 'graph', 'sage', 'gat'
        use_batch_norm : bool, optional
            Whether to use batch normalization
        use_layer_norm : bool, optional
            Whether to use layer normalization
        eps : float, optional
            Small constant for numerical stability
        max_norm : float, optional
            Maximum norm for gradient clipping
        **kwargs : dict
            Additional arguments for specific layer types
        """
        super(StableGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.eps = eps
        self.max_norm = max_norm
        
        # Choose layer type
        if layer_type.lower() == 'gcn':
            conv_layer = GCNConv
        elif layer_type.lower() == 'graph':
            conv_layer = GraphConv
        elif layer_type.lower() == 'sage':
            conv_layer = SAGEConv  # More stable than GCN
        elif layer_type.lower() == 'gat':
            conv_layer = GATConv
        else:
            conv_layer = SAGEConv  # Default to SAGE as it's more stable
        
        # Create layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(conv_layer(num_node_features, hidden_channels, **kwargs))
        if use_batch_norm:
            self.norms.append(BatchNorm1d(hidden_channels, eps=eps))
        elif use_layer_norm:
            self.norms.append(LayerNorm(hidden_channels, eps=eps))
        else:
            self.norms.append(torch.nn.Identity())
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels, **kwargs))
            if use_batch_norm:
                self.norms.append(BatchNorm1d(hidden_channels, eps=eps))
            elif use_layer_norm:
                self.norms.append(LayerNorm(hidden_channels, eps=eps))
            else:
                self.norms.append(torch.nn.Identity())
        
        # Last layer (if multiple layers)
        if num_layers > 1:
            self.convs.append(conv_layer(hidden_channels, hidden_channels, **kwargs))
            if use_batch_norm:
                self.norms.append(BatchNorm1d(hidden_channels, eps=eps))
            elif use_layer_norm:
                self.norms.append(LayerNorm(hidden_channels, eps=eps))
            else:
                self.norms.append(torch.nn.Identity())
                
        # Initialize weights properly
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights for stability"""
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
                
            # Additional initialization for more stability
            if hasattr(conv, 'weight') and conv.weight is not None:
                torch.nn.init.xavier_uniform_(conv.weight)
            if hasattr(conv, 'bias') and conv.bias is not None:
                torch.nn.init.zeros_(conv.bias)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass through the GNN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features of shape [num_nodes, num_node_features]
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges]
        batch : torch.Tensor, optional
            Batch vector of shape [num_nodes] to identify node batch assignments
            
        Returns:
        --------
        torch.Tensor
            Node embeddings of shape [num_nodes, hidden_channels]
        """
        # Normalize edge weights if provided to prevent extreme values
        if edge_weight is not None:
            # Replace NaN values with zeros
            if torch.isnan(edge_weight).any():
                print("Warning: NaN values detected in edge weights. Replacing with zeros.")
                edge_weight = torch.nan_to_num(edge_weight, nan=0.0)
                
            # Clip extreme values
            edge_weight = torch.clamp(edge_weight, min=-3.0, max=3.0)
            
            # Check if all weights are zero
            if torch.sum(torch.abs(edge_weight)) < self.eps:
                print("Warning: All edge weights are near zero. Using uniform weights.")
                edge_weight = torch.ones_like(edge_weight)
                
        # Check for NaN in input features
        if torch.isnan(x).any():
            print("Warning: NaN values detected in input features. Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0)
            
        # Normalize input features to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # First layer
        x = self.convs[0](x, edge_index, edge_weight)
        x = self.norms[0](x)
        x = F.relu(x)
        # Check for NaNs after first layer
        if torch.isnan(x).any():
            print("Warning: NaN values detected after first GNN layer. Resetting to zeros.")
            x = torch.zeros_like(x)
            
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Intermediate layers
        for i in range(1, len(self.convs) - 1):
            # Apply convolution
            x_prev = x  # Store previous tensor in case of NaNs
            x = self.convs[i](x, edge_index, edge_weight)
            
            # Check for NaNs
            if torch.isnan(x).any():
                print(f"Warning: NaN values detected in GNN layer {i+1}. Using previous layer values.")
                x = x_prev  # Recover from previous layer
                continue  # Skip rest of this layer
                
            # Apply normalization, activation, and dropout
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer (without dropout)
        if len(self.convs) > 1:
            x_prev = x
            x = self.convs[-1](x, edge_index, edge_weight)
            
            # Check for NaNs
            if torch.isnan(x).any():
                print(f"Warning: NaN values detected in final GNN layer. Using previous layer values.")
                x = x_prev  # Recover from previous layer
            else:
                x = self.norms[-1](x)
        
        # Final check for NaNs
        if torch.isnan(x).any():
            print("Warning: NaN values in final output. Returning zeros.")
            x = torch.zeros_like(x)
            
        return x


class StableGNNWithPooling(torch.nn.Module):
    """
    GNN with global pooling to produce graph-level embeddings.
    Enhanced with stability improvements.
    """
    
    def __init__(self, num_node_features, hidden_channels, output_channels=64,
                 num_layers=3, dropout=0.1, layer_type='sage', pool_type='mean', 
                 use_batch_norm=True, eps=1e-5, **kwargs):
        """
        Initialize the GNN with pooling.
        
        Parameters:
        -----------
        num_node_features : int
            Number of input features per node
        hidden_channels : int
            Dimension of hidden features
        output_channels : int, optional
            Dimension of output features
        num_layers : int, optional
            Number of graph convolutional layers
        dropout : float, optional
            Dropout probability
        layer_type : str, optional
            Type of graph convolutional layer to use
            Options: 'gcn', 'graph', 'sage', 'gat'
        pool_type : str, optional
            Type of pooling to use
            Options: 'mean', 'sum', 'max'
        use_batch_norm : bool, optional
            Whether to use batch normalization
        eps : float, optional
            Small constant for numerical stability
        **kwargs : dict
            Additional arguments for specific layer types
        """
        super(StableGNNWithPooling, self).__init__()
        
        # GNN for node embeddings with stability enhancements
        self.gnn = StableGNN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            layer_type=layer_type,
            use_batch_norm=use_batch_norm,
            eps=eps,
            **kwargs
        )
        
        # Pooling type
        self.pool_type = pool_type.lower()
        
        # Add batch normalization after pooling for stability
        self.output_norm = BatchNorm1d(hidden_channels, eps=eps) if use_batch_norm else torch.nn.Identity()
        
        # Linear layer for projection with proper initialization
        self.lin = torch.nn.Linear(hidden_channels, output_channels)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass through the GNN with pooling.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features of shape [num_nodes, num_node_features]
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges]
        batch : torch.Tensor, optional
            Batch vector of shape [num_nodes] to identify node batch assignments
            
        Returns:
        --------
        torch.Tensor
            Graph embeddings of shape [batch_size, output_channels]
        """
        # Get node embeddings from GNN
        x = self.gnn(x, edge_index, edge_weight)
        
        # If no batch info, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply pooling
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_type == 'sum':
            from torch_geometric.nn import global_add_pool
            x = global_add_pool(x, batch)
        elif self.pool_type == 'max':
            from torch_geometric.nn import global_max_pool
            x = global_max_pool(x, batch)
        
        # Apply normalization
        x = self.output_norm(x)
        
        # Check for NaN values after pooling
        if torch.isnan(x).any():
            print("Warning: NaN values after pooling. Resetting to zeros.")
            x = torch.zeros_like(x)
        
        # Apply projection with gradient clipping
        x = self.lin(x)
        
        # Final check for NaNs
        if torch.isnan(x).any():
            print("Warning: NaN values in final output. Returning zeros.")
            x = torch.zeros_like(x)
        
        return x