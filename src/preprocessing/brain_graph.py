"""
Brain graph creation from functional connectivity matrices.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def create_brain_graph(connectivity_matrix, roi_features=None, threshold=0.2, 
                       use_absolute=True, self_loops=False):
    """
    Create a PyTorch Geometric graph from a connectivity matrix.
    
    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        Connectivity matrix of shape (n_regions, n_regions)
    roi_features : numpy.ndarray, optional
        Features for each ROI of shape (n_regions, n_features)
    threshold : float, optional
        Threshold for binarizing connectivity matrix
    use_absolute : bool, optional
        Whether to use absolute values of connectivity before thresholding
    self_loops : bool, optional
        Whether to include self-loops in the graph
        
    Returns:
    --------
    torch_geometric.data.Data
        PyTorch Geometric graph
    """
    # Apply threshold to connectivity matrix
    if use_absolute:
        adj_matrix = (np.abs(connectivity_matrix) > threshold).astype(np.float32)
    else:
        adj_matrix = (connectivity_matrix > threshold).astype(np.float32)
    
    # Remove self-loops if requested
    if not self_loops:
        np.fill_diagonal(adj_matrix, 0)
    
    # Get edges (connections between ROIs)
    edges = np.where(adj_matrix > 0)
    edge_index = torch.tensor(np.array([edges[0], edges[1]]), dtype=torch.long)
    
    # Edge weights are the connectivity values
    edge_weights = torch.tensor(connectivity_matrix[edges], dtype=torch.float)
    
    # Create node features
    n_regions = connectivity_matrix.shape[0]
    
    if roi_features is not None:
        # Use provided ROI features
        node_features = torch.tensor(roi_features, dtype=torch.float)
    else:
        # If no ROI features provided, use basic graph measures as features
        # Degree centrality
        degrees = np.sum(adj_matrix, axis=0)
        # Weighted degree centrality
        weighted_degrees = np.sum(np.abs(connectivity_matrix), axis=0)
        
        node_features = torch.tensor(
            np.column_stack([degrees, weighted_degrees]), 
            dtype=torch.float
        )
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features, 
        edge_index=edge_index, 
        edge_attr=edge_weights
    )
    
    return data

def create_dynamic_brain_graphs(dynamic_connectivity, roi_features=None, threshold=0.2):
    """
    Create a list of PyTorch Geometric graphs from dynamic connectivity matrices.
    
    Parameters:
    -----------
    dynamic_connectivity : list
        List of connectivity matrices for each time window
    roi_features : numpy.ndarray, optional
        Features for each ROI of shape (n_regions, n_features)
    threshold : float, optional
        Threshold for binarizing connectivity matrices
        
    Returns:
    --------
    list
        List of PyTorch Geometric graphs
    """
    graphs = []
    
    for conn_matrix in dynamic_connectivity:
        graph = create_brain_graph(conn_matrix, roi_features, threshold)
        graphs.append(graph)
    
    return graphs

class BrainGraphCreator:
    """
    Class to create brain graphs from functional connectivity matrices.
    """
    
    def __init__(self, threshold=0.2, use_absolute=True, self_loops=False,
                 use_dynamic=False):
        """
        Initialize the brain graph creator.
        
        Parameters:
        -----------
        threshold : float, optional
            Threshold for binarizing connectivity matrices
        use_absolute : bool, optional
            Whether to use absolute values of connectivity before thresholding
        self_loops : bool, optional
            Whether to include self-loops in the graph
        use_dynamic : bool, optional
            Whether to create dynamic brain graphs
        """
        self.threshold = threshold
        self.use_absolute = use_absolute
        self.self_loops = self_loops
        self.use_dynamic = use_dynamic
    
    def create_graph(self, features):
        """
        Create a brain graph from extracted features.
        
        Parameters:
        -----------
        features : dict
            Dictionary containing extracted features
            
        Returns:
        --------
        torch_geometric.data.Data or dict
            PyTorch Geometric graph or dictionary containing static and dynamic graphs
        """
        conn_matrix = features['connectivity_matrix']
        roi_features = features['roi_features']
        
        # Create static brain graph
        static_graph = create_brain_graph(
            conn_matrix, 
            roi_features, 
            self.threshold,
            self.use_absolute,
            self.self_loops
        )
        
        if not self.use_dynamic or 'dynamic_connectivity' not in features:
            return static_graph
        
        # Create dynamic brain graphs
        dynamic_graphs = create_dynamic_brain_graphs(
            features['dynamic_connectivity'],
            roi_features,
            self.threshold
        )
        
        return {
            'static_graph': static_graph,
            'dynamic_graphs': dynamic_graphs,
            'window_times': features.get('window_times')
        }
    
    def batch_create_graphs(self, features_list):
        """
        Create brain graphs from a batch of extracted features.
        
        Parameters:
        -----------
        features_list : list
            List of dictionaries containing extracted features
            
        Returns:
        --------
        list
            List of PyTorch Geometric graphs or dictionaries containing static and dynamic graphs
        """
        results = []
        
        for features in tqdm(features_list, desc="Creating brain graphs"):
            graph = self.create_graph(features)
            
            # Add subject ID if available
            if 'subject_id' in features:
                if isinstance(graph, dict):
                    graph['subject_id'] = features['subject_id']
                else:
                    graph.subject_id = features['subject_id']
            
            results.append(graph)
        
        return results