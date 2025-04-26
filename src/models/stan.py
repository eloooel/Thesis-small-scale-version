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
