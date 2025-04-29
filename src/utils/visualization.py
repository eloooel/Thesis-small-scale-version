"""
Visualization utilities for GNN-STAN ADHD classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import torch
from nilearn import plotting


def plot_connectivity_matrix(connectivity_matrix, title='Functional Connectivity Matrix', 
                             cmap='coolwarm', vmin=-1, vmax=1, save_path=None):
    """
    Plot a connectivity matrix.
    
    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        Connectivity matrix to plot
    title : str, optional
        Title of the plot
    cmap : str, optional
        Colormap to use
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(connectivity_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                square=True, xticklabels=False, yticklabels=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_brain_graph(adjacency_matrix, node_positions=None, node_colors=None, 
                     node_size=50, edge_threshold=0.2, title='Brain Graph', save_path=None):
    """
    Plot a brain graph from an adjacency matrix.
    
    Parameters:
    -----------
    adjacency_matrix : numpy.ndarray
        Adjacency matrix of the brain graph
    node_positions : dict, optional
        Dictionary mapping node indices to positions
    node_colors : list, optional
        Colors for each node
    node_size : int or list, optional
        Size of nodes
    edge_threshold : float, optional
        Threshold for displaying edges
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Create a graph from adjacency matrix
    G = nx.from_numpy_array((np.abs(adjacency_matrix) > edge_threshold).astype(int))
    
    # If no positions provided, use spring layout
    if node_positions is None:
        node_positions = nx.spring_layout(G, seed=42)
    
    # Setup colors
    if node_colors is None:
        node_colors = ['skyblue'] * len(G.nodes())
    
    # Plot
    plt.figure(figsize=(12, 10))
    nx.draw_networkx(G, pos=node_positions, node_color=node_colors, 
                     node_size=node_size, edge_color='gray', alpha=0.7,
                     with_labels=False)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_attention_weights(attention_weights, roi_labels=None, 
                           title='Spatial Attention Weights', save_path=None):
    """
    Plot attention weights for brain regions.
    
    Parameters:
    -----------
    attention_weights : numpy.ndarray or torch.Tensor
        Attention weights for brain regions
    roi_labels : list, optional
        Labels for regions of interest
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Convert torch tensor to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Reshape if needed
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    # Create labels if not provided
    if roi_labels is None:
        roi_labels = [f'ROI {i+1}' for i in range(len(attention_weights))]
    
    # Sort by attention weight
    sorted_indices = np.argsort(attention_weights)[::-1]
    sorted_weights = attention_weights[sorted_indices]
    sorted_labels = [roi_labels[i] for i in sorted_indices]
    
    # Plot top 20 regions
    n_regions = min(20, len(sorted_weights))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_regions))
    
    bars = plt.bar(range(n_regions), sorted_weights[:n_regions], color=colors)
    plt.xticks(range(n_regions), sorted_labels[:n_regions], rotation=90)
    plt.ylabel('Attention Weight', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Add weight values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_temporal_attention(temporal_attention, title='Temporal Attention Weights', save_path=None):
    """
    Plot temporal attention weights across time points.
    
    Parameters:
    -----------
    temporal_attention : numpy.ndarray or torch.Tensor
        Temporal attention weights
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Convert torch tensor to numpy if needed
    if isinstance(temporal_attention, torch.Tensor):
        temporal_attention = temporal_attention.detach().cpu().numpy()
    
    # Reshape if needed
    if temporal_attention.ndim > 1:
        # If 2D, use the first row
        temporal_attention = temporal_attention[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(temporal_attention, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.fill_between(range(len(temporal_attention)), 0, temporal_attention, alpha=0.3)
    plt.xlabel('Time Point', fontsize=14)
    plt.ylabel('Attention Weight', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_brain_regions_with_attention(attention_weights, atlas_img, threshold=0.5, 
                                       title='Brain Regions with Attention', save_path=None):
    """
    Plot brain regions colored by attention weights.
    
    Parameters:
    -----------
    attention_weights : numpy.ndarray or torch.Tensor
        Attention weights for brain regions
    atlas_img : str or nibabel.nifti1.Nifti1Image
        Path to atlas image or nibabel image object
    threshold : float, optional
        Threshold for displaying regions
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Convert torch tensor to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Normalize attention weights to [0, 1]
    norm_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    # Create custom colormap for attention
    colors = plt.cm.viridis(norm_weights)
    
    # Plot using nilearn
    fig = plt.figure(figsize=(15, 5))
    
    # Three different views
    for i, view in enumerate(['x', 'y', 'z']):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plotting.plot_roi(atlas_img, bg_img=None, cmap='viridis', 
                         colorbar=True, title=f'{title} - {view} view',
                         axes=ax, view_type=view)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=['Control', 'ADHD'], 
                          normalize=False, title='Confusion Matrix', save_path=None):
    """
    Plot a confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list, optional
        Class names
    normalize : bool, optional
        Whether to normalize the confusion matrix
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_roc_curve(y_true, y_score, title='ROC Curve', save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_precision_recall_curve(y_true, y_score, title='Precision-Recall Curve', save_path=None):
    """
    Plot Precision-Recall curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Compute Precision-Recall curve and PR area
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy'], 
                           title='Training History', save_path=None):
    """
    Plot training history for a deep learning model.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history (e.g., from Keras)
    metrics : list, optional
        List of metrics to plot
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(15, 5 * len(metrics)))
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        
        if f'val_{metric}' in history:
            plt.plot(history[metric], label=f'Training {metric}')
            plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
        else:
            plt.plot(history[metric], label=metric)
            
        plt.title(f'{title} - {metric}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_cross_validation_results(cv_results, metric='accuracy', 
                                  title='Cross-Validation Results', save_path=None):
    """
    Plot cross-validation results.
    
    Parameters:
    -----------
    cv_results : dict or list
        Cross-validation results
    metric : str, optional
        Metric to plot
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Handle different input formats
    if isinstance(cv_results, dict):
        # Scikit-learn style results
        fold_scores = cv_results.get(f'test_{metric}', [])
        fold_labels = [f'Fold {i+1}' for i in range(len(fold_scores))]
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
    elif isinstance(cv_results, list):
        # Custom results format
        fold_scores = [result.get(metric, 0) for result in cv_results]
        fold_labels = [f'Fold {i+1}' for i in range(len(fold_scores))]
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
    else:
        raise ValueError("Unsupported cv_results format")
    
    plt.figure(figsize=(12, 6))
    
    # Plot individual fold scores
    bars = plt.bar(fold_labels, fold_scores, color='skyblue', alpha=0.7)
    
    # Add mean line
    plt.axhline(y=mean_score, color='red', linestyle='-', label=f'Mean {metric}: {mean_score:.3f} ± {std_score:.3f}')
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_site_specific_results(site_results, metrics=['accuracy', 'f1_score'], 
                               title='Site-Specific Results', save_path=None):
    """
    Plot results for each acquisition site in a multi-site study.
    
    Parameters:
    -----------
    site_results : dict
        Dictionary mapping site names to result dictionaries
    metrics : list, optional
        List of metrics to plot
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    sites = list(site_results.keys())
    n_metrics = len(metrics)
    
    plt.figure(figsize=(15, 5 * n_metrics))
    
    for i, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, i+1)
        
        # Extract values for the current metric
        values = [site_results[site].get(metric, 0) for site in sites]
        
        # Plot
        bars = plt.bar(sites, values, color=plt.cm.viridis(np.linspace(0, 1, len(sites))))
        
        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Site', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.title(f'{title} - {metric}', fontsize=16)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_hyperparameter_tuning_results(param_values, scores, param_name,
                                       metric='accuracy', title='Hyperparameter Tuning Results',
                                       save_path=None):
    """
    Plot the effect of a hyperparameter on model performance.
    
    Parameters:
    -----------
    param_values : list
        List of parameter values
    scores : list
        Corresponding performance scores
    param_name : str
        Name of the hyperparameter
    metric : str, optional
        Name of the performance metric
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Line plot
    plt.plot(param_values, scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Highlight best value
    best_idx = np.argmax(scores)
    best_value = param_values[best_idx]
    best_score = scores[best_idx]
    
    plt.scatter([best_value], [best_score], c='red', s=100, zorder=10,
                label=f'Best: {param_name}={best_value}, {metric}={best_score:.3f}')
    
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_feature_importance(feature_importance, feature_names=None,
                            title='Feature Importance', save_path=None):
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    feature_importance : array-like
        Importance scores for features
    feature_names : list, optional
        Names of features
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the figure
    """
    # Convert to numpy array if needed
    if isinstance(feature_importance, torch.Tensor):
        feature_importance = feature_importance.detach().cpu().numpy()
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(feature_importance))]
    
    # Sort by importance
    sorted_idx = np.argsort(feature_importance)
    sorted_importance = feature_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Plot only top 20 features if there are many
    n_features = min(20, len(sorted_importance))
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(n_features), sorted_importance[-n_features:], 
                    color=plt.cm.viridis(np.linspace(0, 1, n_features)))
    plt.yticks(range(n_features), sorted_names[-n_features:])
    plt.xlabel('Importance', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Add importance values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left', va='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()