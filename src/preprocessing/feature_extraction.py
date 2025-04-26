"""
Feature extraction from preprocessed rs-fMRI data.
Contains functions to compute functional connectivity matrices and prepare features for the GNN-STAN model.
"""

import os
import numpy as np
import torch
from nilearn import connectome
from tqdm import tqdm
import warnings

def extract_connectivity_matrix(time_series, kind='correlation'):
    """
    Extract functional connectivity matrix from time series data.
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        Time series data of shape (n_timepoints, n_regions)
    kind : str, optional
        Type of connectivity measure to compute
        Options: 'correlation', 'partial correlation', 'tangent', 'covariance', 'precision'
        
    Returns:
    --------
    numpy.ndarray
        Connectivity matrix of shape (n_regions, n_regions)
    """
    # Ensure time_series is properly shaped
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
    
    # Create connectivity measure
    correlation_measure = connectome.ConnectivityMeasure(kind=kind)
    
    # Compute connectivity matrix
    connectivity_matrix = correlation_measure.fit_transform([time_series])[0]
    
    # Replace NaNs with zeros
    connectivity_matrix = np.nan_to_num(connectivity_matrix, 0)
    
    return connectivity_matrix

def extract_roi_features(time_series):
    """
    Extract features for each ROI from time series data.
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        Time series data of shape (n_timepoints, n_regions)
        
    Returns:
    --------
    numpy.ndarray
        ROI features of shape (n_regions, n_features)
    """
    # Calculate basic statistical features for each ROI
    n_regions = time_series.shape[1]
    roi_features = np.zeros((n_regions, 4))
    
    for i in range(n_regions):
        ts = time_series[:, i]
        # Mean
        roi_features[i, 0] = np.mean(ts)
        # Standard deviation
        roi_features[i, 1] = np.std(ts)
        # Skewness
        roi_features[i, 2] = np.mean(((ts - np.mean(ts)) / np.std(ts)) ** 3) if np.std(ts) > 0 else 0
        # Kurtosis
        roi_features[i, 3] = np.mean(((ts - np.mean(ts)) / np.std(ts)) ** 4) if np.std(ts) > 0 else 0
    
    return roi_features

def create_dynamic_connectivity(time_series, window_size=20, step_size=5):
    """
    Create dynamic functional connectivity matrices using sliding window approach.
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        Time series data of shape (n_timepoints, n_regions)
    window_size : int, optional
        Size of the sliding window
    step_size : int, optional
        Step size between consecutive windows
        
    Returns:
    --------
    list
        List of connectivity matrices for each window
    numpy.ndarray
        Window timestamps
    """
    n_timepoints, n_regions = time_series.shape
    n_windows = (n_timepoints - window_size) // step_size + 1
    
    dynamic_connectivity = []
    window_times = []
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_times.append(start_idx + window_size // 2)  # Middle of the window
        
        # Extract time series for this window
        window_ts = time_series[start_idx:end_idx, :]
        
        # Compute connectivity matrix
        conn_matrix = extract_connectivity_matrix(window_ts)
        dynamic_connectivity.append(conn_matrix)
    
    return dynamic_connectivity, np.array(window_times)

class FeatureExtractor:
    """
    Class to extract features from preprocessed rs-fMRI data.
    """
    
    def __init__(self, connectivity_type='correlation', threshold=0.2, dynamic=False,
                 window_size=20, step_size=5):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        connectivity_type : str, optional
            Type of connectivity measure to compute
        threshold : float, optional
            Threshold for binarizing connectivity matrices
        dynamic : bool, optional
            Whether to compute dynamic connectivity matrices
        window_size : int, optional
            Size of the sliding window for dynamic connectivity
        step_size : int, optional
            Step size between consecutive windows for dynamic connectivity
        """
        self.connectivity_type = connectivity_type
        self.threshold = threshold
        self.dynamic = dynamic
        self.window_size = window_size
        self.step_size = step_size
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
    
    def extract_features(self, time_series, return_dynamic=False):
        """
        Extract features from preprocessed time series data.
        
        Parameters:
        -----------
        time_series : numpy.ndarray
            Time series data of shape (n_timepoints, n_regions)
        return_dynamic : bool, optional
            Whether to return dynamic connectivity matrices
            
        Returns:
        --------
        dict
            Dictionary containing extracted features
        """
        # Extract static functional connectivity matrix
        conn_matrix = extract_connectivity_matrix(time_series, kind=self.connectivity_type)
        
        # Extract ROI features
        roi_features = extract_roi_features(time_series)
        
        result = {
            'connectivity_matrix': conn_matrix,
            'roi_features': roi_features
        }
        
        # Compute dynamic connectivity if requested
        if self.dynamic:
            dynamic_conn, window_times = create_dynamic_connectivity(
                time_series, 
                window_size=self.window_size,
                step_size=self.step_size
            )
            
            result['dynamic_connectivity'] = dynamic_conn
            result['window_times'] = window_times
        
        return result
    
    def batch_extract_features(self, data_list):
        """
        Extract features from a batch of preprocessed data.
        
        Parameters:
        -----------
        data_list : list
            List of preprocessed data dictionaries or numpy arrays
            
        Returns:
        --------
        list
            List of dictionaries containing extracted features
        """
        results = []
        
        for i, data in enumerate(tqdm(data_list, desc="Extracting features")):
            if isinstance(data, dict) and 'time_series' in data:
                time_series = data['time_series']
                subject_id = data.get('subject_id', f"subject_{i}")
            elif isinstance(data, str) and os.path.exists(data):
                # Assume it's a path to a .npy file
                time_series = np.load(data)
                subject_id = os.path.basename(data).split('_')[0]
            elif isinstance(data, np.ndarray):
                time_series = data
                subject_id = f"subject_{i}"
            else:
                print(f"Warning: Unrecognized data format for item {i}. Skipping.")
                continue
                
            # Extract features
            features = self.extract_features(time_series)
            features['subject_id'] = subject_id
            
            results.append(features)
        
        return results