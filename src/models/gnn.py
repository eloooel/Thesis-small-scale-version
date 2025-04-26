"""
Graph Neural Network (GNN) module for ADHD classification.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv, global_mean_pool

class GNN(torch.nn.Module):
    """
    Graph Neural Network component of the hybrid model.
    
    This module implements a configurable GNN with multiple graph convolutional layers
    and various supported layer types.
    """
    
    def __init__(self, num_node_features, hidden_channels, num_layers=3, 
                 dropout=0.1, layer_type='gcn', **kwargs):
        """
        Initialize the GNN module.
        
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
        **kwargs : dict
            Additional arguments for specific layer types
        """
        super(GNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Choose layer type
        conv_layer = {
            'gcn': GCNConv,
            'graph': GraphConv,
            'sage': SAGEConv,
            'gat': GATConv
        }.get(layer_type.lower(), GCNConv)
        
        # Create layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(conv_layer(num_node_features, hidden_channels, **kwargs))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels, **kwargs))
        
        # Last layer (if multiple layers)
        if num_layers > 1:
            self.convs.append(conv_layer(hidden_channels, hidden_channels, **kwargs))
    
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
        # First layer
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Intermediate layers
        for i in range(1, len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer (without dropout)
        if len(self.convs) > 1:
            x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

class GNNWithPooling(torch.nn.Module):
    """
    GNN with global pooling to produce graph-level embeddings.
    """
    
    def __init__(self, num_node_features, hidden_channels, output_channels=64,
                 num_layers=3, dropout=0.1, layer_type='gcn', pool_type='mean', **kwargs):
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
        **kwargs : dict
            Additional arguments for specific layer types
        """
        super(GNNWithPooling, self).__init__()
        
        # GNN for node embeddings
        self.gnn = GNN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            layer_type=layer_type,
            **kwargs
        )
        
        # Pooling type
        self.pool_type = pool_type.lower()
        
        # Linear layer for projection
        self.lin = torch.nn.Linear(hidden_channels, output_channels)
    
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
        
        # Apply projection
        x = self.lin(x)
        
        return x