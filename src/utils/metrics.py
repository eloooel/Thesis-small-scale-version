"""
Evaluation metrics for ADHD classification.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix

def calculate_binary_metrics(y_true, y_pred, y_score=None, threshold=0.5):
    """
    Calculate metrics for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    y_score : array-like, optional
        Predicted probabilities or scores
    threshold : float, optional
        Threshold for converting scores to binary predictions
        
    Returns:
    --------
    dict
        Dictionary containing various metrics
    """
    # If y_score is provided but y_pred is not, convert scores to predictions
    if y_score is not None and y_pred is None:
        y_pred = (np.array(y_score) >= threshold).astype(int)
    
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_score is not None and isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure shapes are correct (flatten if needed)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if y_score is not None:
        y_score = np.array(y_score).flatten()
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate ROC AUC if scores are provided
    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except:
            # Handle case where only one class is present
            roc_auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }

def calculate_cross_validation_metrics(cv_results):
    """
    Calculate average metrics across cross-validation folds.
    
    Parameters:
    -----------
    cv_results : list
        List of dictionaries containing metrics for each fold
        
    Returns:
    --------
    dict
        Dictionary containing average metrics and standard deviations
    """
    # Metrics to aggregate
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
    
    # Initialize dictionaries for means and standard deviations
    means = {}
    stds = {}
    
    # Calculate mean and standard deviation for each metric
    for metric in metrics:
        values = [result[metric] for result in cv_results if metric in result and result[metric] is not None]
        
        if values:
            means[metric] = np.mean(values)
            stds[metric] = np.std(values)
        else:
            means[metric] = None
            stds[metric] = None
    
    # Aggregate confusion matrices
    cm_sum = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    for result in cv_results:
        if 'confusion_matrix' in result:
            cm = result['confusion_matrix']
            for key in cm_sum:
                cm_sum[key] += cm[key]
    
    return {
        'means': means,
        'stds': stds,
        'confusion_matrix_sum': cm_sum
    }

def print_metrics_summary(metrics):
    """
    Print a summary of classification metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics
    """
    if 'means' in metrics:
        # Cross-validation results
        means = metrics['means']
        stds = metrics['stds']
        
        print("\n=== Cross-Validation Results ===")
        print(f"Accuracy: {means['accuracy']:.4f} ± {stds['accuracy']:.4f}")
        print(f"Precision: {means['precision']:.4f} ± {stds['precision']:.4f}")
        print(f"Recall: {means['recall']:.4f} ± {stds['recall']:.4f}")
        print(f"F1-Score: {means['f1_score']:.4f} ± {stds['f1_score']:.4f}")
        print(f"Specificity: {means['specificity']:.4f} ± {stds['specificity']:.4f}")
        
        if means['roc_auc'] is not None:
            print(f"ROC AUC: {means['roc_auc']:.4f} ± {stds['roc_auc']:.4f}")
        
        cm = metrics['confusion_matrix_sum']
        print("\nConfusion Matrix (Sum):")
        print(f"TN: {cm['tn']}, FP: {cm['fp']}")
        print(f"FN: {cm['fn']}, TP: {cm['tp']}")
    else:
        # Single result
        print("\n=== Classification Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print("\nConfusion Matrix:")
        print(f"TN: {cm['tn']}, FP: {cm['fp']}")
        print(f"FN: {cm['fn']}, TP: {cm['tp']}")