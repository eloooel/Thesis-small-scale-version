"""
Hybrid GNN-STAN model for ADHD classification.
Combines Graph Neural Networks (GNN) with Spatio-Temporal Attention Networks (STAN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnn import GNN, GNNWithPooling
from .stan import STAN, DynamicSTAN

class GNN_STAN(nn.Module):
    """
    Hybrid GNN-STAN model for ADHD classification from rs-fMRI data.
    
    This model combines the strengths of GNNs for modeling brain connectivity
    with STAN for capturing spatial and temporal attention patterns.
    """
    
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
    
    def forward(self, data, return_attention=False):
        """
        Forward pass through the GNN-STAN model.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data or dict
            Input data
            If torch_geometric.data.Data: static graph
            If dict: dynamic graphs with keys 'static_graph' and 'dynamic_graphs'
        return_attention : bool, optional
            Whether to return attention weights
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape [batch_size, output_dim]
        dict, optional
            Attention weights if return_attention=True
        """
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
                
                # Final classification
                output = self.classifier(stan_out)
                
                return output, attention_weights
            else:
                stan_out = self.stan(dynamic_node_features)
                
                # Final classification
                output = self.classifier(stan_out)
                
                return output
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
            
            # Process through STAN
            if return_attention:
                stan_out, attention_weights = self.stan(
                    node_features,
                    batch=data.batch,
                    return_attention=True
                )
                
                # Final classification
                output = self.classifier(stan_out)
                
                return output, attention_weights
            else:
                stan_out = self.stan(node_features, batch=data.batch)
                
                # Final classification
                output = self.classifier(stan_out)
                
                return output

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
    
    def forward(self, data, return_attention=False, return_logits=False):
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
            
        Returns:
        --------
        torch.Tensor
            Output probabilities or logits of shape [batch_size, 1]
        dict, optional
            Attention weights if return_attention=True
        """
        if return_attention:
            logits, attention_weights = self.model(data, return_attention=True)
            
            if return_logits:
                return logits, attention_weights
            else:
                probs = torch.sigmoid(logits)
                return probs, attention_weights
        else:
            logits = self.model(data)
            
            if return_logits:
                return logits
            else:
                probs = torch.sigmoid(logits)
                return probs
    
    def predict(self, data, threshold=0.5):
        """
        Make binary predictions.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data or dict
            Input data
        threshold : float, optional
            Classification threshold
            
        Returns:
        --------
        torch.Tensor
            Binary predictions of shape [batch_size, 1]
        """
        probs = self.forward(data)
        return (probs >= threshold).float()