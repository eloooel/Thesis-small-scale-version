#!/usr/bin/env python
"""
Main entry point for running GNN-STAN ADHD classification experiments.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.config import Config, create_experiment_config
from src.data import ADHD200Dataset
from src.preprocessing import Preprocessor, FeatureExtractor, BrainGraphCreator
from src.models import GNN_STAN_Classifier
from src.utils.metrics import calculate_binary_metrics, calculate_cross_validation_metrics, print_metrics_summary


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config):
    """Setup logging for the experiment."""
    log_file = os.path.join(config.output_dir, f"{config.experiment_name}.log")
    log_level = logging.INFO if config.verbose else logging.WARNING
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_model(model, data_loader, criterion, optimizer, device, epoch, logger):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        
        # Compute loss
        labels = batch.y.float()
        loss = criterion(output.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
    
    return avg_loss


def validate_model(model, data_loader, criterion, device, epoch, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Compute loss
            labels = batch.y.float()
            loss = criterion(output.squeeze(), labels)
            
            total_loss += loss.item()
            
            # Store predictions
            all_labels.append(labels.cpu().numpy())
            pred = (output.squeeze() > 0.5).float()
            all_preds.append(pred.cpu().numpy())
            all_scores.append(output.squeeze().cpu().numpy())
    
    # Concatenate results
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    
    # Compute metrics
    metrics = calculate_binary_metrics(all_labels, all_preds, all_scores)
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch+1}: Val Loss = {avg_loss:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    return avg_loss, metrics


def run_experiment(config, logger):
    """Run an experiment based on the provided configuration."""
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = ADHD200Dataset(
        data_dir=config.data.data_dir,
        n_subjects=config.data.n_subjects,
        download=config.data.download,
        atlas=config.data.atlas,
        test_size=config.data.test_size,
        random_state=config.data.random_seed
    )
    
    # Setup components based on configuration
    logger.info("Setting up preprocessing pipeline...")
    preprocessor = Preprocessor(
        output_dir=config.preprocessing.output_dir,
        tr=config.preprocessing.tr
    )
    
    feature_extractor = FeatureExtractor(
        connectivity_type=config.feature_extraction.connectivity_type,
        threshold=config.feature_extraction.threshold,
        dynamic=config.feature_extraction.dynamic,
        window_size=config.feature_extraction.window_size,
        step_size=config.feature_extraction.step_size
    )
    
    graph_creator = BrainGraphCreator(
        threshold=config.brain_graph.threshold,
        use_absolute=config.brain_graph.use_absolute,
        self_loops=config.brain_graph.self_loops,
        use_dynamic=config.brain_graph.use_dynamic
    )
    
    # Run the experiment based on the cross-validation strategy
    if config.evaluation.cv_strategy == 'kfold':
        logger.info(f"Running {config.evaluation.n_folds}-fold cross-validation...")
        folds = dataset.create_cv_folds(
            n_folds=config.evaluation.n_folds, 
            stratify_by=config.evaluation.stratify_by
        )
        run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                            graph_creator, folds, logger)
        
    elif config.evaluation.cv_strategy == 'loso':
        logger.info("Running Leave-One-Site-Out cross-validation...")
        folds = dataset.create_loso_folds()
        run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                            graph_creator, folds, logger)
        
    else:
        logger.info("Running train/test split evaluation...")
        train_indices, test_indices = dataset.create_train_test_split(
            stratify_by=config.evaluation.stratify_by
        )
        run_train_test_evaluation(config, dataset, preprocessor, feature_extractor, 
                                 graph_creator, train_indices, test_indices, logger)
    
    logger.info(f"Experiment {config.experiment_name} completed.")


def run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                        graph_creator, folds, logger):
    """Run cross-validation experiment."""
    cv_results = []
    
    for fold, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"Processing fold {fold+1}/{len(folds)}")
        
        # Prepare data for this fold
        logger.info("Preparing data...")
        train_graphs, train_labels = dataset.prepare_data_for_model(
            indices=train_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        val_graphs, val_labels = dataset.prepare_data_for_model(
            indices=val_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        # Check if CUDA is available
        device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create and train model
        logger.info("Creating model...")
        
        # Get model parameters from the first graph
        if isinstance(train_graphs[0], dict):
            graph = train_graphs[0]['static_graph']
            num_time_points = len(train_graphs[0]['dynamic_graphs'])
        else:
            graph = train_graphs[0]
            num_time_points = config.model.num_time_points
        
        num_node_features = graph.x.shape[1]
        
        # Update model config with data-dependent parameters
        config.model.num_node_features = num_node_features
        config.model.num_time_points = num_time_points
        
        # Initialize model
        model = GNN_STAN_Classifier(
            num_node_features=num_node_features,
            hidden_channels=config.model.hidden_channels,
            num_time_points=num_time_points,
            gnn_layers=config.model.gnn_layers,
            gnn_type=config.model.gnn_type,
            use_edge_weights=config.model.use_edge_weights,
            dropout=config.model.dropout,
            use_dynamic=config.model.use_dynamic,
            bidirectional=config.model.bidirectional
        ).to(device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train the model
        logger.info("Training model...")
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        criterion = torch.nn.BCELoss()
        
        # Move graphs to device
        train_graphs_device = []
        for graph in train_graphs:
            if isinstance(graph, dict):
                graph_device = {
                    'static_graph': graph['static_graph'].to(device),
                    'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                }
                train_graphs_device.append(graph_device)
            else:
                train_graphs_device.append(graph.to(device))
        
        val_graphs_device = []
        for graph in val_graphs:
            if isinstance(graph, dict):
                graph_device = {
                    'static_graph': graph['static_graph'].to(device),
                    'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                }
                val_graphs_device.append(graph_device)
            else:
                val_graphs_device.append(graph.to(device))
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.training.num_epochs):
            # Train and validate
            train_loss = 0
            val_loss = 0
            val_metrics = None
            
            # Evaluation on validation set
            model.eval()
            with torch.no_grad():
                all_val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
                all_val_outputs = []
                
                for i, graph in enumerate(val_graphs_device):
                    output = model(graph)
                    all_val_outputs.append(output.squeeze())
                
                all_val_outputs = torch.stack(all_val_outputs)
                val_loss = criterion(all_val_outputs, all_val_labels)
                
                # Predictions
                val_preds = (all_val_outputs > 0.5).float()
                
                # Calculate metrics
                val_metrics = calculate_binary_metrics(
                    all_val_labels.cpu().numpy(), 
                    val_preds.cpu().numpy(),
                    all_val_outputs.cpu().numpy()
                )
            
            # Training
            model.train()
            for i, graph in enumerate(train_graphs_device):
                optimizer.zero_grad()
                output = model(graph)
                label = torch.tensor([float(train_labels[i])], dtype=torch.float).to(device)
                loss = criterion(output.squeeze(), label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Calculate average loss
            train_loss /= len(train_graphs_device)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss.item():.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val F1 Score: {val_metrics['f1_score']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if config.training.save_model:
                    model_save_path = os.path.join(
                        config.training.model_save_dir,
                        f"{config.experiment_name}_fold{fold+1}_best.pt"
                    )
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"  Saved best model to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluate best model on validation set
        if config.training.save_model:
            model_save_path = os.path.join(
                config.training.model_save_dir,
                f"{config.experiment_name}_fold{fold+1}_best.pt"
            )
            model.load_state_dict(torch.load(model_save_path))
        
        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            all_val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
            all_val_outputs = []
            
            for i, graph in enumerate(val_graphs_device):
                output = model(graph)
                all_val_outputs.append(output.squeeze())
            
            all_val_outputs = torch.stack(all_val_outputs)
            
            # Predictions
            val_preds = (all_val_outputs > 0.5).float()
            
            # Calculate metrics
            val_metrics = calculate_binary_metrics(
                all_val_labels.cpu().numpy(), 
                val_preds.cpu().numpy(),
                all_val_outputs.cpu().numpy()
            )
        
        # Log validation results
        logger.info(f"Fold {fold+1} validation results:")
        for metric, value in val_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Store results for this fold
        cv_results.append(val_metrics)
    
    # Calculate cross-validation summary
    cv_summary = calculate_cross_validation_metrics(cv_results)
    
    # Log cross-validation summary
    logger.info("Cross-validation summary:")
    print_metrics_summary(cv_summary)
    
    # Save summary to file
    summary_file = os.path.join(config.output_dir, f"{config.experiment_name}_summary.txt")
    with open(summary_file, 'w') as f:
        for metric, value in cv_summary['means'].items():
            f.write(f"{metric}_mean: {value}\n")
            f.write(f"{metric}_std: {cv_summary['stds'][metric]}\n")
        
        f.write("\nConfusion Matrix (Sum):\n")
        cm = cv_summary['confusion_matrix_sum']
        f.write(f"TN: {cm['tn']}, FP: {cm['fp']}\n")
        f.write(f"FN: {cm['fn']}, TP: {cm['tp']}\n")
    
    logger.info(f"Summary saved to {summary_file}")


def run_train_test_evaluation(config, dataset, preprocessor, feature_extractor, 
                             graph_creator, train_indices, test_indices, logger):
    """Run train/test split evaluation."""
    # Prepare data
    logger.info("Preparing data...")
    train_graphs, train_labels = dataset.prepare_data_for_model(
        indices=train_indices,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        graph_creator=graph_creator
    )
    
    test_graphs, test_labels = dataset.prepare_data_for_model(
        indices=test_indices,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        graph_creator=graph_creator
    )
    
    # Check if CUDA is available
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create and train model
    logger.info("Creating model...")
    
    # Get model parameters from the first graph
    if isinstance(train_graphs[0], dict):
        graph = train_graphs[0]['static_graph']
        num_time_points = len(train_graphs[0]['dynamic_graphs'])
    else:
        graph = train_graphs[0]
        num_time_points = config.model.num_time_points
    
    num_node_features = graph.x.shape[1]
    
    # Update model config with data-dependent parameters
    config.model.num_node_features = num_node_features
    config.model.num_time_points = num_time_points
    
    # Initialize model
    model = GNN_STAN_Classifier(
        num_node_features=num_node_features,
        hidden_channels=config.model.hidden_channels,
        num_time_points=num_time_points,
        gnn_layers=config.model.gnn_layers,
        gnn_type=config.model.gnn_type,
        use_edge_weights=config.model.use_edge_weights,
        dropout=config.model.dropout,
        use_dynamic=config.model.use_dynamic,
        bidirectional=config.model.bidirectional
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    criterion = torch.nn.BCELoss()
    
    # Class weights for imbalanced data
    if config.training.use_class_weights:
        pos_weight = np.sum(train_labels == 0) / np.sum(train_labels == 1)
        logger.info(f"Using class weights: positive weight = {pos_weight:.2f}")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Move graphs to device
    train_graphs_device = []
    for graph in train_graphs:
        if isinstance(graph, dict):
            graph_device = {
                'static_graph': graph['static_graph'].to(device),
                'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
            }
            train_graphs_device.append(graph_device)
        else:
            train_graphs_device.append(graph.to(device))
    
    test_graphs_device = []
    for graph in test_graphs:
        if isinstance(graph, dict):
            graph_device = {
                'static_graph': graph['static_graph'].to(device),
                'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
            }
            test_graphs_device.append(graph_device)
        else:
            test_graphs_device.append(graph.to(device))
    
    # Training loop
    logger.info("Training model...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for i, graph in enumerate(train_graphs_device):
            optimizer.zero_grad()
            output = model(graph)
            label = torch.tensor([float(train_labels[i])], dtype=torch.float).to(device)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_graphs_device)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, graph in enumerate(test_graphs_device):
                output = model(graph)
                label = torch.tensor([float(test_labels[i])], dtype=torch.float).to(device)
                loss = criterion(output.squeeze(), label)
                val_loss += loss.item()
        
        val_loss /= len(test_graphs_device)
        val_losses.append(val_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if config.training.save_model:
                model_save_path = os.path.join(
                    config.training.model_save_dir,
                    f"{config.experiment_name}_best.pt"
                )
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"  Saved best model to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(config.output_dir, f"{config.experiment_name}_loss.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Loss plot saved to {loss_plot_path}")
    
    # Load best model for evaluation
    if config.training.save_model:
        model_save_path = os.path.join(
            config.training.model_save_dir,
            f"{config.experiment_name}_best.pt"
        )
        model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.eval()
    
    test_preds = []
    test_scores = []
    
    with torch.no_grad():
        for graph in test_graphs_device:
            output = model(graph)
            test_scores.append(output.squeeze().cpu().numpy())
            pred = (output.squeeze() > config.evaluation.threshold).float().cpu().numpy()
            test_preds.append(pred)
    
    test_preds = np.array(test_preds)
    test_scores = np.array(test_scores)
    
    # Calculate metrics
    metrics = calculate_binary_metrics(test_labels, test_preds, test_scores)
    
    # Log test results
    logger.info("Test results:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(config.output_dir, f"{config.experiment_name}_metrics.txt")
    with open(metrics_file, 'w') as f:
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value}\n")
        
        f.write("\nConfusion Matrix:\n")
        cm = metrics['confusion_matrix']
        f.write(f"TN: {cm['tn']}, FP: {cm['fp']}\n")
        f.write(f"FN: {cm['fn']}, TP: {cm['tp']}\n")
    
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Extract and visualize attention weights
    if config.model.use_dynamic:
        logger.info("Extracting attention weights...")
        
        with torch.no_grad():
            # Use the first test subject for visualization
            graph = test_graphs_device[0]
            output, attention_weights = model(graph, return_attention=True)
            
            # Save attention weights
            if 'spatial_weights' in attention_weights:
                spatial_weights = attention_weights['spatial_weights']
                
                if isinstance(spatial_weights, torch.Tensor):
                    # Save spatial attention weights plot
                    from src.utils.visualization import plot_attention_weights
                    
                    attention_plot_path = os.path.join(
                        config.output_dir, 
                        f"{config.experiment_name}_spatial_attention.png"
                    )
                    
                    # Get atlas labels if available
                    atlas_data = dataset.get_atlas()
                    roi_labels = atlas_data.get('labels', None)
                    
                    # Plot and save
                    plt.figure(figsize=(12, 8))
                    plot_attention_weights(
                        spatial_weights.cpu().numpy(), 
                        roi_labels=roi_labels,
                        title='Spatial Attention Weights',
                        save_path=attention_plot_path
                    )
                    
                    logger.info(f"Spatial attention weights plot saved to {attention_plot_path}")
            
            if 'temporal_weights' in attention_weights:
                temporal_weights = attention_weights['temporal_weights']
                
                if isinstance(temporal_weights, torch.Tensor):
                    # Save temporal attention weights plot
                    from src.utils.visualization import plot_temporal_attention
                    
                    temporal_plot_path = os.path.join(
                        config.output_dir, 
                        f"{config.experiment_name}_temporal_attention.png"
                    )
                    
                    # Plot and save
                    plt.figure(figsize=(12, 6))
                    plot_temporal_attention(
                        temporal_weights.cpu().numpy(),
                        title='Temporal Attention Weights',
                        save_path=temporal_plot_path
                    )
                    
                    logger.info(f"Temporal attention weights plot saved to {temporal_plot_path}")


def run_hyperparameter_tuning(config, logger):
    """Run hyperparameter tuning experiment."""
    logger.info(f"Starting hyperparameter tuning experiment: {config.experiment_name}")
    
    if not config.hyperparameter_tuning:
        logger.error("No hyperparameters specified for tuning. Please update the configuration.")
        return
    
    # Load dataset
    dataset = ADHD200Dataset(
        data_dir=config.data.data_dir,
        n_subjects=config.data.n_subjects,
        download=config.data.download,
        atlas=config.data.atlas,
        test_size=config.data.test_size,
        random_state=config.data.random_seed
    )
    
    # Create cross-validation folds
    folds = dataset.create_cv_folds(
        n_folds=config.evaluation.n_folds, 
        stratify_by=config.evaluation.stratify_by
    )
    
    # Setup components
    preprocessor = Preprocessor(
        output_dir=config.preprocessing.output_dir,
        tr=config.preprocessing.tr
    )
    
    feature_extractor = FeatureExtractor(
        connectivity_type=config.feature_extraction.connectivity_type,
        threshold=config.feature_extraction.threshold,
        dynamic=config.feature_extraction.dynamic,
        window_size=config.feature_extraction.window_size,
        step_size=config.feature_extraction.step_size
    )
    
    graph_creator = BrainGraphCreator(
        threshold=config.brain_graph.threshold,
        use_absolute=config.brain_graph.use_absolute,
        self_loops=config.brain_graph.self_loops,
        use_dynamic=config.brain_graph.use_dynamic
    )
    
    # Prepare data (do this once to save time)
    logger.info("Preparing data for all folds...")
    all_data = {}
    
    for fold, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"Preparing data for fold {fold+1}/{len(folds)}")
        
        train_graphs, train_labels = dataset.prepare_data_for_model(
            indices=train_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        val_graphs, val_labels = dataset.prepare_data_for_model(
            indices=val_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        all_data[fold] = {
            'train_graphs': train_graphs,
            'train_labels': train_labels,
            'val_graphs': val_graphs,
            'val_labels': val_labels
        }
    
    # Extract parameters from the first graph
    if isinstance(all_data[0]['train_graphs'][0], dict):
        graph = all_data[0]['train_graphs'][0]['static_graph']
        num_time_points = len(all_data[0]['train_graphs'][0]['dynamic_graphs'])
    else:
        graph = all_data[0]['train_graphs'][0]
        num_time_points = config.model.num_time_points
    
    num_node_features = graph.x.shape[1]
    
    # Update model config
    config.model.num_node_features = num_node_features
    config.model.num_time_points = num_time_points
    
    # Generate hyperparameter combinations
    import itertools
    
    param_names = list(config.hyperparameter_tuning.keys())
    param_values = list(config.hyperparameter_tuning.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Running {len(param_combinations)} hyperparameter combinations")
    
    # Store results
    tuning_results = []
    
    # Check if CUDA is available
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run each hyperparameter combination
    for i, combination in enumerate(param_combinations):
        logger.info(f"Combination {i+1}/{len(param_combinations)}: {dict(zip(param_names, combination))}")
        
        # Create a new config for this combination
        param_config = dict(zip(param_names, combination))
        
        # Update config
        for param, value in param_config.items():
            if param.startswith('model.'):
                param_path = param.split('.')
                setattr(config.model, param_path[1], value)
            elif param.startswith('training.'):
                param_path = param.split('.')
                setattr(config.training, param_path[1], value)
            elif param.startswith('feature_extraction.'):
                param_path = param.split('.')
                setattr(config.feature_extraction, param_path[1], value)
            elif param.startswith('brain_graph.'):
                param_path = param.split('.')
                setattr(config.brain_graph, param_path[1], value)
            else:
                logger.warning(f"Unknown parameter: {param}")
        
        # Run cross-validation for this combination
        cv_results = []
        
        for fold, (train_indices, val_indices) in enumerate(folds):
            logger.info(f"Processing fold {fold+1}/{len(folds)}")
            
            # Get data for this fold
            train_graphs = all_data[fold]['train_graphs']
            train_labels = all_data[fold]['train_labels']
            val_graphs = all_data[fold]['val_graphs']
            val_labels = all_data[fold]['val_labels']
            
            # Move data to device
            train_graphs_device = []
            for graph in train_graphs:
                if isinstance(graph, dict):
                    graph_device = {
                        'static_graph': graph['static_graph'].to(device),
                        'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                    }
                    train_graphs_device.append(graph_device)
                else:
                    train_graphs_device.append(graph.to(device))
            
            val_graphs_device = []
            for graph in val_graphs:
                if isinstance(graph, dict):
                    graph_device = {
                        'static_graph': graph['static_graph'].to(device),
                        'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                    }
                    val_graphs_device.append(graph_device)
                else:
                    val_graphs_device.append(graph.to(device))
            
            # Initialize model with current hyperparameters
            model = GNN_STAN_Classifier(
                num_node_features=num_node_features,
                hidden_channels=config.model.hidden_channels,
                num_time_points=num_time_points,
                gnn_layers=config.model.gnn_layers,
                gnn_type=config.model.gnn_type,
                use_edge_weights=config.model.use_edge_weights,
                dropout=config.model.dropout,
                use_dynamic=config.model.use_dynamic,
                bidirectional=config.model.bidirectional
            ).to(device)
            
            # Define optimizer and loss function
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.training.learning_rate, 
                weight_decay=config.training.weight_decay
            )
            criterion = torch.nn.BCELoss()
            
            # Training loop
            best_val_metrics = None
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(config.training.num_epochs):
                # Training
                model.train()
                train_loss = 0
                
                for i, graph in enumerate(train_graphs_device):
                    optimizer.zero_grad()
                    output = model(graph)
                    label = torch.tensor([float(train_labels[i])], dtype=torch.float).to(device)
                    loss = criterion(output.squeeze(), label)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_graphs_device)
                
                # Validation
                model.eval()
                val_loss = 0
                val_outputs = []
                val_preds = []
                
                with torch.no_grad():
                    for i, graph in enumerate(val_graphs_device):
                        output = model(graph)
                        label = torch.tensor([float(val_labels[i])], dtype=torch.float).to(device)
                        loss = criterion(output.squeeze(), label)
                        val_loss += loss.item()
                        val_outputs.append(output.squeeze().cpu().numpy())
                        pred = (output.squeeze() > config.evaluation.threshold).float().cpu().numpy()
                        val_preds.append(pred)
                
                val_loss /= len(val_graphs_device)
                
                # Calculate metrics
                val_metrics = calculate_binary_metrics(
                    val_labels, 
                    np.array(val_preds), 
                    np.array(val_outputs)
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_metrics = val_metrics
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.training.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Store results for this fold
            cv_results.append(best_val_metrics)
        
        # Calculate average metrics across folds
        cv_summary = calculate_cross_validation_metrics(cv_results)
        
        # Store results for this combination
        result = {
            'params': param_config,
            'cv_summary': cv_summary
        }
        tuning_results.append(result)
        
        # Log results
        logger.info(f"Combination {i+1} results:")
        for param, value in param_config.items():
            logger.info(f"  {param}: {value}")
        
        for metric, value in cv_summary['means'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f} ± {cv_summary['stds'][metric]:.4f}")
    
    # Find best combination
    best_score = -float('inf')
    best_combination = None
    best_result = None
    target_metric = 'f1_score'  # Use F1 score as the target metric
    
    for result in tuning_results:
        score = result['cv_summary']['means'][target_metric]
        if score > best_score:
            best_score = score
            best_combination = result['params']
            best_result = result
    
    # Log best results
    logger.info(f"Best hyperparameter combination (by {target_metric}):")
    for param, value in best_combination.items():
        logger.info(f"  {param}: {value}")
    
    logger.info(f"Best {target_metric}: {best_score:.4f}")
    
    # Save results to file
    results_file = os.path.join(config.output_dir, f"{config.experiment_name}_hyperparameter_tuning.txt")
    with open(results_file, 'w') as f:
        f.write(f"Best hyperparameter combination (by {target_metric}):\n")
        for param, value in best_combination.items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nBest {target_metric}: {best_score:.4f}\n")
        
        f.write("\nAll results:\n")
        for i, result in enumerate(tuning_results):
            f.write(f"\nCombination {i+1}:\n")
            for param, value in result['params'].items():
                f.write(f"  {param}: {value}\n")
            
            for metric, value in result['cv_summary']['means'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f} ± {result['cv_summary']['stds'][metric]:.4f}\n")
    
    logger.info(f"Hyperparameter tuning results saved to {results_file}")
    
    # Save hyperparameter tuning plots
    for param in param_names:
        if len(config.hyperparameter_tuning[param]) > 1:
            # Get all results for this parameter
            param_values = []
            param_scores = []
            
            for value in config.hyperparameter_tuning[param]:
                # Find all results with this parameter value
                matching_results = [
                    result for result in tuning_results 
                    if result['params'].get(param) == value
                ]
                
                if matching_results:
                    # Calculate average score
                    avg_score = np.mean([
                        result['cv_summary']['means'][target_metric] 
                        for result in matching_results
                    ])
                    
                    param_values.append(value)
                    param_scores.append(avg_score)
            
            # Plot parameter vs. score
            from src.utils.visualization import plot_hyperparameter_tuning_results
            
            plot_path = os.path.join(
                config.output_dir, 
                f"{config.experiment_name}_param_{param.replace('.', '_')}.png"
            )
            
            plt.figure(figsize=(10, 6))
            plot_hyperparameter_tuning_results(
                param_values, 
                param_scores, 
                param_name=param,
                metric=target_metric,
                title=f"Effect of {param} on {target_metric}",
                save_path=plot_path
            )
            
            logger.info(f"Parameter tuning plot saved to {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GNN-STAN ADHD Classification')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (mini_test, full_dynamic, static_only, hyperparameter_tuning)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (overrides experiment name)')
    parser.add_argument('--subjects', type=int, default=None,
                        help='Number of subjects to use (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--folds', type=int, default=None,
                        help='Number of cross-validation folds (overrides config)')
    parser.add_argument('--cv', type=str, default=None, choices=['kfold', 'loso'],
                        help='Cross-validation strategy (overrides config)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device index to use (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load config from file
        config = Config.from_yaml(args.config)
    else:
        # Create config based on experiment name
        config = create_experiment_config(args.experiment)
    
    # Override config with command line arguments
    if args.subjects:
        config.data.n_subjects = args.subjects
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    if args.folds:
        config.evaluation.n_folds = args.folds
    
    if args.cv:
        config.evaluation.cv_strategy = args.cv
    
    if args.gpu is not None:
        if args.gpu >= 0:
            config.training.device = f'cuda:{args.gpu}'
        else:
            config.training.device = 'cpu'
    
    if args.seed:
        config.random_seed = args.seed
        config.data.random_seed = args.seed
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(config.random_seed)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Log configuration
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Save configuration
    config.to_yaml(os.path.join(config.output_dir, f"{config.experiment_name}_config.yaml"))
    
    # Run the appropriate experiment
    try:
        if args.experiment == 'hyperparameter_tuning':
            run_hyperparameter_tuning(config, logger)
        else:
            run_experiment(config, logger)
    except Exception as e:
        logger.exception(f"Error during experiment: {e}")
        raise
    
    logger.info("Done!")


if __name__ == '__main__':
    main()#!/usr/bin/env python
"""
Main entry point for running GNN-STAN ADHD classification experiments.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.config import Config, create_experiment_config
from src.data import ADHD200Dataset
from src.preprocessing import Preprocessor, FeatureExtractor, BrainGraphCreator
from src.models import GNN_STAN_Classifier
from src.utils.metrics import calculate_binary_metrics, calculate_cross_validation_metrics, print_metrics_summary


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config):
    """Setup logging for the experiment."""
    log_file = os.path.join(config.output_dir, f"{config.experiment_name}.log")
    log_level = logging.INFO if config.verbose else logging.WARNING
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_model(model, data_loader, criterion, optimizer, device, epoch, logger):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        
        # Compute loss
        labels = batch.y.float()
        loss = criterion(output.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
    
    return avg_loss


def validate_model(model, data_loader, criterion, device, epoch, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Compute loss
            labels = batch.y.float()
            loss = criterion(output.squeeze(), labels)
            
            total_loss += loss.item()
            
            # Store predictions
            all_labels.append(labels.cpu().numpy())
            pred = (output.squeeze() > 0.5).float()
            all_preds.append(pred.cpu().numpy())
            all_scores.append(output.squeeze().cpu().numpy())
    
    # Concatenate results
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    
    # Compute metrics
    metrics = calculate_binary_metrics(all_labels, all_preds, all_scores)
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch+1}: Val Loss = {avg_loss:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    return avg_loss, metrics


def run_experiment(config, logger):
    """Run an experiment based on the provided configuration."""
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = ADHD200Dataset(
        data_dir=config.data.data_dir,
        n_subjects=config.data.n_subjects,
        download=config.data.download,
        atlas=config.data.atlas,
        test_size=config.data.test_size,
        random_state=config.data.random_seed
    )
    
    # Setup components based on configuration
    logger.info("Setting up preprocessing pipeline...")
    preprocessor = Preprocessor(
        output_dir=config.preprocessing.output_dir,
        tr=config.preprocessing.tr
    )
    
    feature_extractor = FeatureExtractor(
        connectivity_type=config.feature_extraction.connectivity_type,
        threshold=config.feature_extraction.threshold,
        dynamic=config.feature_extraction.dynamic,
        window_size=config.feature_extraction.window_size,
        step_size=config.feature_extraction.step_size
    )
    
    graph_creator = BrainGraphCreator(
        threshold=config.brain_graph.threshold,
        use_absolute=config.brain_graph.use_absolute,
        self_loops=config.brain_graph.self_loops,
        use_dynamic=config.brain_graph.use_dynamic
    )
    
    # Run the experiment based on the cross-validation strategy
    if config.evaluation.cv_strategy == 'kfold':
        logger.info(f"Running {config.evaluation.n_folds}-fold cross-validation...")
        folds = dataset.create_cv_folds(
            n_folds=config.evaluation.n_folds, 
            stratify_by=config.evaluation.stratify_by
        )
        run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                            graph_creator, folds, logger)
        
    elif config.evaluation.cv_strategy == 'loso':
        logger.info("Running Leave-One-Site-Out cross-validation...")
        folds = dataset.create_loso_folds()
        run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                            graph_creator, folds, logger)
        
    else:
        logger.info("Running train/test split evaluation...")
        train_indices, test_indices = dataset.create_train_test_split(
            stratify_by=config.evaluation.stratify_by
        )
        run_train_test_evaluation(config, dataset, preprocessor, feature_extractor, 
                                 graph_creator, train_indices, test_indices, logger)
    
    logger.info(f"Experiment {config.experiment_name} completed.")


def run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                        graph_creator, folds, logger):
    """Run cross-validation experiment."""
    cv_results = []
    
    for fold, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"Processing fold {fold+1}/{len(folds)}")
        
        # Prepare data for this fold
        logger.info("Preparing data...")
        train_graphs, train_labels = dataset.prepare_data_for_model(
            indices=train_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        val_graphs, val_labels = dataset.prepare_data_for_model(
            indices=val_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        # Check if CUDA is available
        device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create and train model
        logger.info("Creating model...")
        
        # Get model parameters from the first graph
        if isinstance(train_graphs[0], dict):
            graph = train_graphs[0]['static_graph']
            num_time_points = len(train_graphs[0]['dynamic_graphs'])
        else:
            graph = train_graphs[0]
            num_time_points = config.model.num_time_points
        
        num_node_features = graph.x.shape[1]
        
        # Update model config with data-dependent parameters
        config.model.num_node_features = num_node_features
        config.model.num_time_points = num_time_points
        
        # Initialize model
        model = GNN_STAN_Classifier(
            num_node_features=num_node_features,
            hidden_channels=config.model.hidden_channels,
            num_time_points=num_time_points,
            gnn_layers=config.model.gnn_layers,
            gnn_type=config.model.gnn_type,
            use_edge_weights=config.model.use_edge_weights,
            dropout=config.model.dropout,
            use_dynamic=config.model.use_dynamic,
            bidirectional=config.model.bidirectional
        ).to(device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train the model
        logger.info("Training model...")
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        criterion = torch.nn.BCELoss()
        
        # Move graphs to device
        train_graphs_device = []
        for graph in train_graphs:
            if isinstance(graph, dict):
                graph_device = {
                    'static_graph': graph['static_graph'].to(device),
                    'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                }
                train_graphs_device.append(graph_device)
            else:
                train_graphs_device.append(graph.to(device))
        
        val_graphs_device = []
        for graph in val_graphs:
            if isinstance(graph, dict):
                graph_device = {
                    'static_graph': graph['static_graph'].to(device),
                    'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                }
                val_graphs_device.append(graph_device)
            else:
                val_graphs_device.append(graph.to(device))
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.training.num_epochs):
            # Train and validate
            train_loss = 0
            val_loss = 0
            val_metrics = None
            
            # Evaluation on validation set
            model.eval()
            with torch.no_grad():
                all_val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
                all_val_outputs = []
                
                for i, graph in enumerate(val_graphs_device):
                    output = model(graph)
                    all_val_outputs.append(output.squeeze())
                
                all_val_outputs = torch.stack(all_val_outputs)
                val_loss = criterion(all_val_outputs, all_val_labels)
                
                # Predictions
                val_preds = (all_val_outputs > 0.5).float()
                
                # Calculate metrics
                val_metrics = calculate_binary_metrics(
                    all_val_labels.cpu().numpy(), 
                    val_preds.cpu().numpy(),
                    all_val_outputs.cpu().numpy()
                )
            
            # Training
            model.train()
            for i, graph in enumerate(train_graphs_device):
                optimizer.zero_grad()
                output = model(graph)
                label = torch.tensor([float(train_labels[i])], dtype=torch.float).to(device)
                loss = criterion(output.squeeze(), label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Calculate average loss
            train_loss /= len(train_graphs_device)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss.item():.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val F1 Score: {val_metrics['f1_score']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if config.training.save_model:
                    model_save_path = os.path.join(
                        config.training.model_save_dir,
                        f"{config.experiment_name}_fold{fold+1}_best.pt"
                    )
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"  Saved best model to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluate best model on validation set
        if config.training.save_model:
            model_save_path = os.path.join(
                config.training.model_save_dir,
                f"{config.experiment_name}_fold{fold+1}_best.pt"
            )
            model.load_state_dict(torch.load(model_save_path))
        
        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            all_val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
            all_val_outputs = []
            
            for i, graph in enumerate(val_graphs_device):
                output = model(graph)
                all_val_outputs.append(output.squeeze())
            
            all_val_outputs = torch.stack(all_val_outputs)
            
            # Predictions
            val_preds = (all_val_outputs > 0.5).float()
            
            # Calculate metrics
            val_metrics = calculate_binary_metrics(
                all_val_labels.cpu().numpy(), 
                val_preds.cpu().numpy(),
                all_val_outputs.cpu().numpy()
            )
        
        # Log validation results
        logger.info(f"Fold {fold+1} validation results:")
        for metric, value in val_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Store results for this fold
        cv_results.append(val_metrics)
    
    # Calculate cross-validation summary
    cv_summary = calculate_cross_validation_metrics(cv_results)
    
    # Log cross-validation summary
    logger.info("Cross-validation summary:")
    print_metrics_summary(cv_summary)
    
    # Save summary to file
    summary_file = os.path.join(config.output_dir, f"{config.experiment_name}_summary.txt")
    with open(summary_file, 'w') as f:
        for metric, value in cv_summary['means'].items():
            f.write(f"{metric}_mean: {value}\n")
            f.write(f"{metric}_std: {cv_summary['stds'][metric]}\n")
        
        f.write("\nConfusion Matrix (Sum):\n")
        cm = cv_summary['confusion_matrix_sum']
        f.write(f"TN: {cm['tn']}, FP: {cm['fp']}\n")
        f.write(f"FN: {cm['fn']}, TP: {cm['tp']}\n")
    
    logger.info(f"Summary saved to {summary_file}")


def run_train_test_evaluation(config, dataset, preprocessor, feature_extractor, 
                             graph_creator, train_indices, test_indices, logger):
    """Run train/test split evaluation."""
    # Prepare data
    logger.info("Preparing data...")
    train_graphs, train_labels = dataset.prepare_data_for_model(
        indices=train_indices,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        graph_creator=graph_creator
    )
    
    test_graphs, test_labels = dataset.prepare_data_for_model(
        indices=test_indices,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        graph_creator=graph_creator
    )
    
    # Check if CUDA is available
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create and train model
    logger.info("Creating model...")
    
    # Get model parameters from the first graph
    if isinstance(train_graphs[0], dict):
        graph = train_graphs[0]['static_graph']
        num_time_points = len(train_graphs[0]['dynamic_graphs'])
    else:
        graph = train_graphs[0]
        num_time_points = config.model.num_time_points
    
    num_node_features = graph.x.shape[1]
    
    # Update model config with data-dependent parameters
    config.model.num_node_features = num_node_features
    config.model.num_time_points = num_time_points
    
    # Initialize model
    model = GNN_STAN_Classifier(
        num_node_features=num_node_features,
        hidden_channels=config.model.hidden_channels,
        num_time_points=num_time_points,
        gnn_layers=config.model.gnn_layers,
        gnn_type=config.model.gnn_type,
        use_edge_weights=config.model.use_edge_weights,
        dropout=config.model.dropout,
        use_dynamic=config.model.use_dynamic,
        bidirectional=config.model.bidirectional
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    criterion = torch.nn.BCELoss()
    
    # Class weights for imbalanced data
    if config.training.use_class_weights:
        pos_weight = np.sum(train_labels == 0) / np.sum(train_labels == 1)
        logger.info(f"Using class weights: positive weight = {pos_weight:.2f}")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Move graphs to device
    train_graphs_device = []
    for graph in train_graphs:
        if isinstance(graph, dict):
            graph_device = {
                'static_graph': graph['static_graph'].to(device),
                'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
            }
            train_graphs_device.append(graph_device)
        else:
            train_graphs_device.append(graph.to(device))
    
    test_graphs_device = []
    for graph in test_graphs:
        if isinstance(graph, dict):
            graph_device = {
                'static_graph': graph['static_graph'].to(device),
                'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
            }
            test_graphs_device.append(graph_device)
        else:
            test_graphs_device.append(graph.to(device))
    
    # Training loop
    logger.info("Training model...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for i, graph in enumerate(train_graphs_device):
            optimizer.zero_grad()
            output = model(graph)
            label = torch.tensor([float(train_labels[i])], dtype=torch.float).to(device)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_graphs_device)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, graph in enumerate(test_graphs_device):
                output = model(graph)
                label = torch.tensor([float(test_labels[i])], dtype=torch.float).to(device)
                loss = criterion(output.squeeze(), label)
                val_loss += loss.item()
        
        val_loss /= len(test_graphs_device)
        val_losses.append(val_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if config.training.save_model:
                model_save_path = os.path.join(
                    config.training.model_save_dir,
                    f"{config.experiment_name}_best.pt"
                )
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"  Saved best model to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(config.output_dir, f"{config.experiment_name}_loss.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Loss plot saved to {loss_plot_path}")
    
    # Load best model for evaluation
    if config.training.save_model:
        model_save_path = os.path.join(
            config.training.model_save_dir,
            f"{config.experiment_name}_best.pt"
        )
        model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.eval()
    
    test_preds = []
    test_scores = []
    
    with torch.no_grad():
        for graph in test_graphs_device:
            output = model(graph)
            test_scores.append(output.squeeze().cpu().numpy())
            pred = (output.squeeze() > config.evaluation.threshold).float().cpu().numpy()
            test_preds.append(pred)
    
    test_preds = np.array(test_preds)
    test_scores = np.array(test_scores)
    
    # Calculate metrics
    metrics = calculate_binary_metrics(test_labels, test_preds, test_scores)
    
    # Log test results
    logger.info("Test results:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(config.output_dir, f"{config.experiment_name}_metrics.txt")
    with open(metrics_file, 'w') as f:
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value}\n")
        
        f.write("\nConfusion Matrix:\n")
        cm = metrics['confusion_matrix']
        f.write(f"TN: {cm['tn']}, FP: {cm['fp']}\n")
        f.write(f"FN: {cm['fn']}, TP: {cm['tp']}\n")
    
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Extract and visualize attention weights
    if config.model.use_dynamic:
        logger.info("Extracting attention weights...")
        
        with torch.no_grad():
            # Use the first test subject for visualization
            graph = test_graphs_device[0]
            output, attention_weights = model(graph, return_attention=True)
            
            # Save attention weights
            if 'spatial_weights' in attention_weights:
                spatial_weights = attention_weights['spatial_weights']
                
                if isinstance(spatial_weights, torch.Tensor):
                    # Save spatial attention weights plot
                    from src.utils.visualization import plot_attention_weights
                    
                    attention_plot_path = os.path.join(
                        config.output_dir, 
                        f"{config.experiment_name}_spatial_attention.png"
                    )
                    
                    # Get atlas labels if available
                    atlas_data = dataset.get_atlas()
                    roi_labels = atlas_data.get('labels', None)
                    
                    # Plot and save
                    plt.figure(figsize=(12, 8))
                    plot_attention_weights(
                        spatial_weights.cpu().numpy(), 
                        roi_labels=roi_labels,
                        title='Spatial Attention Weights',
                        save_path=attention_plot_path
                    )
                    
                    logger.info(f"Spatial attention weights plot saved to {attention_plot_path}")
            
            if 'temporal_weights' in attention_weights:
                temporal_weights = attention_weights['temporal_weights']
                
                if isinstance(temporal_weights, torch.Tensor):
                    # Save temporal attention weights plot
                    from src.utils.visualization import plot_temporal_attention
                    
                    temporal_plot_path = os.path.join(
                        config.output_dir, 
                        f"{config.experiment_name}_temporal_attention.png"
                    )
                    
                    # Plot and save
                    plt.figure(figsize=(12, 6))
                    plot_temporal_attention(
                        temporal_weights.cpu().numpy(),
                        title='Temporal Attention Weights',
                        save_path=temporal_plot_path
                    )
                    
                    logger.info(f"Temporal attention weights plot saved to {temporal_plot_path}")


def run_hyperparameter_tuning(config, logger):
    """Run hyperparameter tuning experiment."""
    logger.info(f"Starting hyperparameter tuning experiment: {config.experiment_name}")
    
    if not config.hyperparameter_tuning:
        logger.error("No hyperparameters specified for tuning. Please update the configuration.")
        return
    
    # Load dataset
    dataset = ADHD200Dataset(
        data_dir=config.data.data_dir,
        n_subjects=config.data.n_subjects,
        download=config.data.download,
        atlas=config.data.atlas,
        test_size=config.data.test_size,
        random_state=config.data.random_seed
    )
    
    # Create cross-validation folds
    folds = dataset.create_cv_folds(
        n_folds=config.evaluation.n_folds, 
        stratify_by=config.evaluation.stratify_by
    )
    
    # Setup components
    preprocessor = Preprocessor(
        output_dir=config.preprocessing.output_dir,
        tr=config.preprocessing.tr
    )
    
    feature_extractor = FeatureExtractor(
        connectivity_type=config.feature_extraction.connectivity_type,
        threshold=config.feature_extraction.threshold,
        dynamic=config.feature_extraction.dynamic,
        window_size=config.feature_extraction.window_size,
        step_size=config.feature_extraction.step_size
    )
    
    graph_creator = BrainGraphCreator(
        threshold=config.brain_graph.threshold,
        use_absolute=config.brain_graph.use_absolute,
        self_loops=config.brain_graph.self_loops,
        use_dynamic=config.brain_graph.use_dynamic
    )
    
    # Prepare data (do this once to save time)
    logger.info("Preparing data for all folds...")
    all_data = {}
    
    for fold, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"Preparing data for fold {fold+1}/{len(folds)}")
        
        train_graphs, train_labels = dataset.prepare_data_for_model(
            indices=train_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        val_graphs, val_labels = dataset.prepare_data_for_model(
            indices=val_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        all_data[fold] = {
            'train_graphs': train_graphs,
            'train_labels': train_labels,
            'val_graphs': val_graphs,
            'val_labels': val_labels
        }
    
    # Extract parameters from the first graph
    if isinstance(all_data[0]['train_graphs'][0], dict):
        graph = all_data[0]['train_graphs'][0]['static_graph']
        num_time_points = len(all_data[0]['train_graphs'][0]['dynamic_graphs'])
    else:
        graph = all_data[0]['train_graphs'][0]
        num_time_points = config.model.num_time_points
    
    num_node_features = graph.x.shape[1]
    
    # Update model config
    config.model.num_node_features = num_node_features
    config.model.num_time_points = num_time_points
    
    # Generate hyperparameter combinations
    import itertools
    
    param_names = list(config.hyperparameter_tuning.keys())
    param_values = list(config.hyperparameter_tuning.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Running {len(param_combinations)} hyperparameter combinations")
    
    # Store results
    tuning_results = []#!/usr/bin/env python
"""
Main entry point for running GNN-STAN ADHD classification experiments.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.config import Config, create_experiment_config
from src.data import ADHD200Dataset
from src.preprocessing import Preprocessor, FeatureExtractor, BrainGraphCreator
from src.models import GNN_STAN_Classifier
from src.utils.metrics import calculate_binary_metrics, calculate_cross_validation_metrics, print_metrics_summary


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config):
    """Setup logging for the experiment."""
    log_file = os.path.join(config.output_dir, f"{config.experiment_name}.log")
    log_level = logging.INFO if config.verbose else logging.WARNING
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_model(model, data_loader, criterion, optimizer, device, epoch, logger):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        
        # Compute loss
        labels = batch.y.float()
        loss = criterion(output.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
    
    return avg_loss


def validate_model(model, data_loader, criterion, device, epoch, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Compute loss
            labels = batch.y.float()
            loss = criterion(output.squeeze(), labels)
            
            total_loss += loss.item()
            
            # Store predictions
            all_labels.append(labels.cpu().numpy())
            pred = (output.squeeze() > 0.5).float()
            all_preds.append(pred.cpu().numpy())
            all_scores.append(output.squeeze().cpu().numpy())
    
    # Concatenate results
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    
    # Compute metrics
    metrics = calculate_binary_metrics(all_labels, all_preds, all_scores)
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch {epoch+1}: Val Loss = {avg_loss:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    return avg_loss, metrics


def run_experiment(config, logger):
    """Run an experiment based on the provided configuration."""
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = ADHD200Dataset(
        data_dir=config.data.data_dir,
        n_subjects=config.data.n_subjects,
        download=config.data.download,
        atlas=config.data.atlas,
        test_size=config.data.test_size,
        random_state=config.data.random_seed
    )
    
    # Setup components based on configuration
    logger.info("Setting up preprocessing pipeline...")
    preprocessor = Preprocessor(
        output_dir=config.preprocessing.output_dir,
        tr=config.preprocessing.tr
    )
    
    feature_extractor = FeatureExtractor(
        connectivity_type=config.feature_extraction.connectivity_type,
        threshold=config.feature_extraction.threshold,
        dynamic=config.feature_extraction.dynamic,
        window_size=config.feature_extraction.window_size,
        step_size=config.feature_extraction.step_size
    )
    
    graph_creator = BrainGraphCreator(
        threshold=config.brain_graph.threshold,
        use_absolute=config.brain_graph.use_absolute,
        self_loops=config.brain_graph.self_loops,
        use_dynamic=config.brain_graph.use_dynamic
    )
    
    # Run the experiment based on the cross-validation strategy
    if config.evaluation.cv_strategy == 'kfold':
        logger.info(f"Running {config.evaluation.n_folds}-fold cross-validation...")
        folds = dataset.create_cv_folds(
            n_folds=config.evaluation.n_folds, 
            stratify_by=config.evaluation.stratify_by
        )
        run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                            graph_creator, folds, logger)
        
    elif config.evaluation.cv_strategy == 'loso':
        logger.info("Running Leave-One-Site-Out cross-validation...")
        folds = dataset.create_loso_folds()
        run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                            graph_creator, folds, logger)
        
    else:
        logger.info("Running train/test split evaluation...")
        train_indices, test_indices = dataset.create_train_test_split(
            stratify_by=config.evaluation.stratify_by
        )
        run_train_test_evaluation(config, dataset, preprocessor, feature_extractor, 
                                 graph_creator, train_indices, test_indices, logger)
    
    logger.info(f"Experiment {config.experiment_name} completed.")


def run_cross_validation(config, dataset, preprocessor, feature_extractor, 
                        graph_creator, folds, logger):
    """Run cross-validation experiment."""
    cv_results = []
    
    for fold, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"Processing fold {fold+1}/{len(folds)}")
        
        # Prepare data for this fold
        logger.info("Preparing data...")
        train_graphs, train_labels = dataset.prepare_data_for_model(
            indices=train_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        val_graphs, val_labels = dataset.prepare_data_for_model(
            indices=val_indices,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            graph_creator=graph_creator
        )
        
        # Check if CUDA is available
        device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create and train model
        logger.info("Creating model...")
        
        # Get model parameters from the first graph
        if isinstance(train_graphs[0], dict):
            graph = train_graphs[0]['static_graph']
            num_time_points = len(train_graphs[0]['dynamic_graphs'])
        else:
            graph = train_graphs[0]
            num_time_points = config.model.num_time_points
        
        num_node_features = graph.x.shape[1]
        
        # Update model config with data-dependent parameters
        config.model.num_node_features = num_node_features
        config.model.num_time_points = num_time_points
        
        # Initialize model
        model = GNN_STAN_Classifier(
            num_node_features=num_node_features,
            hidden_channels=config.model.hidden_channels,
            num_time_points=num_time_points,
            gnn_layers=config.model.gnn_layers,
            gnn_type=config.model.gnn_type,
            use_edge_weights=config.model.use_edge_weights,
            dropout=config.model.dropout,
            use_dynamic=config.model.use_dynamic,
            bidirectional=config.model.bidirectional
        ).to(device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train the model
        logger.info("Training model...")
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        criterion = torch.nn.BCELoss()
        
        # Move graphs to device
        train_graphs_device = []
        for graph in train_graphs:
            if isinstance(graph, dict):
                graph_device = {
                    'static_graph': graph['static_graph'].to(device),
                    'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                }
                train_graphs_device.append(graph_device)
            else:
                train_graphs_device.append(graph.to(device))
        
        val_graphs_device = []
        for graph in val_graphs:
            if isinstance(graph, dict):
                graph_device = {
                    'static_graph': graph['static_graph'].to(device),
                    'dynamic_graphs': [g.to(device) for g in graph['dynamic_graphs']]
                }
                val_graphs_device.append(graph_device)
            else:
                val_graphs_device.append(graph.to(device))
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.training.num_epochs):
            # Train and validate
            train_loss = 0
            val_loss = 0
            val_metrics = None
            
            # Evaluation on validation set
            model.eval()
            with torch.no_grad():
                all_val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
                all_val_outputs = []
                
                for i, graph in enumerate(val_graphs_device):
                    output = model(graph)