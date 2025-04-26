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