"""
Data loader for local ADHD dataset (Peking_1).
"""

import os
import numpy as np
import pandas as pd
import glob
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from nilearn import datasets


class LocalADHDDataset:
    """
    Class to handle loading and preprocessing of a local ADHD dataset.
    
    This class provides functionality to:
    1. Load data from a local directory
    2. Extract phenotypic information
    3. Split data into train/test sets
    4. Create cross-validation folds
    5. Preprocess the data
    """
    
    def __init__(self, data_dir, atlas='schaefer_200', test_size=0.2, random_state=42):
        """
        Initialize the local ADHD dataset.
        
        Parameters:
        -----------
        data_dir : str
            Path to the local dataset directory
        atlas : str, optional
            Atlas to use for parcellation ('schaefer_200', 'aal116', etc.)
        test_size : float, optional
            Fraction of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.atlas = atlas
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize empty attributes
        self.func_files = []
        self.anat_files = []
        self.subject_ids = []
        self.sites = []
        self.labels = []  # Will be filled manually or from metadata
        self.train_indices = None
        self.test_indices = None
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Load the local ADHD dataset.
        
        This function scans the provided directory for NIfTI files and extracts subject information.
        """
        print(f"Loading local ADHD dataset from {self.data_dir}")
        
        # Find all functional MRI files
        func_pattern = os.path.join(self.data_dir, "sub-*", "ses-*", "func", "*_task-rest_*.nii.gz")
        self.func_files = sorted(glob.glob(func_pattern))
        
        if not self.func_files:
            raise FileNotFoundError(f"No functional MRI files found in {self.data_dir}")
        
        # Find all anatomical MRI files
        anat_pattern = os.path.join(self.data_dir, "sub-*", "ses-*", "anat", "*_T1w.nii.gz")
        self.anat_files = sorted(glob.glob(anat_pattern))
        
        # Extract subject IDs from filenames
        self.subject_ids = []
        for func_file in self.func_files:
            # Extract subject ID from the file path
            # Assuming path format: .../sub-XXXXXXX/ses-X/...
            parts = os.path.normpath(func_file).split(os.sep)
            for part in parts:
                if part.startswith("sub-"):
                    self.subject_ids.append(part)
                    break
        
        # Set site information (assuming all from same site, e.g., 'Peking')
        self.sites = ['Peking' for _ in self.func_files]
        
        # Try to load labels from metadata file if it exists
        metadata_file = os.path.join(self.data_dir, "participants.tsv")
        if os.path.exists(metadata_file):
            try:
                metadata = pd.read_csv(metadata_file, sep='\t')
                # Assuming the metadata file has 'participant_id' and 'diagnosis' columns
                # where diagnosis is 1 for ADHD and 0 for control
                self.labels = []
                for subject_id in self.subject_ids:
                    # Remove 'sub-' prefix if present in metadata
                    subject_id_clean = subject_id.replace('sub-', '')
                    # Get diagnosis for this subject
                    diagnosis = metadata.loc[metadata['participant_id'] == subject_id_clean, 'diagnosis'].values
                    if len(diagnosis) > 0:
                        self.labels.append(diagnosis[0])
                    else:
                        # Default to 0 (control) if not found
                        print(f"Warning: No diagnosis found for {subject_id}. Defaulting to 0 (control).")
                        self.labels.append(0)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                # Initialize with placeholder labels (all 0)
                self.labels = [0] * len(self.func_files)
        else:
            print("Warning: No metadata file found. Using placeholder labels (all 0).")
            # Initialize with placeholder labels (all 0)
            self.labels = [0] * len(self.func_files)
            
            # Manually set labels here based on your knowledge of the dataset
            # For example:
            # self.labels[0] = 1  # First subject has ADHD
            # self.labels[1] = 0  # Second subject is control
            
            # For Peking_1 dataset, if you know the labels:
            # Assuming format of subject IDs: 10xxxxx
            # - 10xxxxx where last digit is odd: typically ADHD
            # - 10xxxxx where last digit is even: typically control
            # (This is just an example - please adjust based on your actual dataset)
            for i, subject_id in enumerate(self.subject_ids):
                # Extract the numeric part (remove 'sub-' prefix)
                subject_num = subject_id.replace('sub-', '')
                try:
                    # Check if last digit is odd (ADHD) or even (control)
                    if int(subject_num[-1]) % 2 == 1:
                        self.labels[i] = 1  # ADHD
                    else:
                        self.labels[i] = 0  # Control
                except:
                    # Keep default 0 if conversion fails
                    pass
        
        print(f"Loaded dataset with {len(self.func_files)} subjects")
        print(f"Subject IDs: {self.subject_ids}")
        print(f"Labels: {self.labels}")
    
    def create_train_test_split(self, stratify_by='label'):
        """
        Create train/test split.
        
        Parameters:
        -----------
        stratify_by : str, optional
            How to stratify the split ('label', 'site', or None)
        
        Returns:
        --------
        tuple
            (train_indices, test_indices)
        """
        indices = np.arange(len(self.labels))
        
        if stratify_by == 'label':
            stratify = self.labels
        elif stratify_by == 'site':
            stratify = self.sites
        else:
            stratify = None
        
        self.train_indices, self.test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        print(f"Created train/test split: {len(self.train_indices)} train, {len(self.test_indices)} test")
        
        return self.train_indices, self.test_indices
    
    def create_cv_folds(self, n_folds=5, stratify_by='label'):
        """
        Create cross-validation folds.
        
        Parameters:
        -----------
        n_folds : int, optional
            Number of folds
        stratify_by : str, optional
            How to stratify the folds ('label', 'site', or None)
        
        Returns:
        --------
        list
            List of (train_indices, val_indices) tuples for each fold
        """
        indices = np.arange(len(self.labels))
        
        if stratify_by == 'label':
            stratify = self.labels
        elif stratify_by == 'site':
            stratify = self.sites
        else:
            stratify = None
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        folds = []
        for train_idx, val_idx in skf.split(indices, stratify):
            folds.append((train_idx, val_idx))
        
        print(f"Created {n_folds} cross-validation folds")
        
        return folds
    
    def create_loso_folds(self):
        """
        Create Leave-One-Site-Out cross-validation folds.
        
        Returns:
        --------
        list
            List of (train_indices, val_indices) tuples for each site
        """
        unique_sites = np.unique(self.sites)
        
        folds = []
        for site in unique_sites:
            # Test indices are subjects from the current site
            val_indices = np.where(np.array(self.sites) == site)[0]
            
            # Train indices are subjects from all other sites
            train_indices = np.where(np.array(self.sites) != site)[0]
            
            folds.append((train_indices, val_indices))
        
        print(f"Created {len(unique_sites)} Leave-One-Site-Out folds")
        
        return folds
    
    def get_atlas(self):
        """
        Get the atlas for parcellation.
        
        Returns:
        --------
        dict
            Dictionary containing atlas information
        """
        if self.atlas == 'schaefer_200':
            # Schaefer 200 ROI atlas (7 networks)
            atlas_data = datasets.fetch_atlas_schaefer_2018(
                n_rois=200, 
                yeo_networks=7, 
                resolution_mm=2
            )
            return {
                'maps': atlas_data.maps,
                'labels': atlas_data.labels,
                'description': '200 ROIs from Schaefer 2018 atlas (7 networks)'
            }
        elif self.atlas == 'aal116':
            # AAL 116 ROI atlas
            atlas_data = datasets.fetch_atlas_aal()
            return {
                'maps': atlas_data.maps,
                'labels': atlas_data.indices,
                'description': '116 ROIs from AAL atlas'
            }
        else:
            raise ValueError(f"Unsupported atlas: {self.atlas}")
    
    def preprocess_subject(self, index, preprocessor=None):
        """
        Preprocess a single subject.
        
        Parameters:
        -----------
        index : int
            Subject index
        preprocessor : src.preprocessing.Preprocessor, optional
            Preprocessor instance to use (if None, a simplified approach is used)
        
        Returns:
        --------
        dict
            Dictionary containing preprocessed data
        """
        func_file = self.func_files[index]
        subject_id = self.subject_ids[index]
        
        if preprocessor is not None:
            # Use the provided preprocessor
            result = preprocessor.preprocess_fmri(func_file, subject_id)
        else:
            # Use a simplified approach with nilearn
            from nilearn import image, masking
            
            # Load the NIfTI file
            print(f"Loading data for subject {subject_id}...")
            img = nib.load(func_file)
            
            # Basic preprocessing
            print("Applying basic preprocessing...")
            img = image.clean_img(
                img,
                detrend=True,
                standardize='zscore',
                low_pass=0.08,
                high_pass=0.009,
                t_r=2.0  # Default TR
            )
            
            # Create a brain mask
            print("Creating brain mask...")
            mask_img = masking.compute_epi_mask(img)
            
            # Extract time series using the atlas
            print("Extracting ROI time series...")
            atlas_data = self.get_atlas()
            masker = NiftiLabelsMasker(
                labels_img=atlas_data['maps'], 
                standardize=True,
                memory='nilearn_cache', 
                verbose=0
            )
            time_series = masker.fit_transform(img)
            
            # Create result dictionary
            result = {
                'subject_id': subject_id,
                'time_series': time_series,
                'mask': mask_img,
                'label': self.labels[index],
                'site': self.sites[index]
            }
            
        return result
    
    def batch_preprocess(self, indices, preprocessor=None):
        """
        Preprocess a batch of subjects.
        
        Parameters:
        -----------
        indices : list
            List of subject indices
        preprocessor : src.preprocessing.Preprocessor, optional
            Preprocessor instance to use (if None, a simplified approach is used)
        
        Returns:
        --------
        list
            List of dictionaries containing preprocessed data
        """
        results = []
        for i, index in enumerate(indices):
            print(f"\nProcessing subject {i+1}/{len(indices)} (index {index})")
            result = self.preprocess_subject(index, preprocessor)
            results.append(result)
        
        return results
    
    def extract_features(self, preprocessed_data, feature_extractor):
        """
        Extract features from preprocessed data.
        
        Parameters:
        -----------
        preprocessed_data : list
            List of dictionaries containing preprocessed data
        feature_extractor : src.preprocessing.FeatureExtractor
            Feature extractor instance
        
        Returns:
        --------
        list
            List of dictionaries containing extracted features
        """
        # Extract time series data
        time_series_data = [data['time_series'] for data in preprocessed_data]
        
        # Extract features
        features = feature_extractor.batch_extract_features(time_series_data)
        
        # Add labels and subject IDs
        for i, (feature, data) in enumerate(zip(features, preprocessed_data)):
            feature['label'] = data.get('label', self.labels[i])
            feature['subject_id'] = data.get('subject_id', f"subject_{i}")
            feature['site'] = data.get('site', None)
        
        return features
    
    def create_brain_graphs(self, features, graph_creator):
        """
        Create brain graphs from extracted features.
        
        Parameters:
        -----------
        features : list
            List of dictionaries containing extracted features
        graph_creator : src.preprocessing.BrainGraphCreator
            Brain graph creator instance
        
        Returns:
        --------
        list
            List of PyTorch Geometric Data objects or dictionaries containing static and dynamic graphs
        """
        return graph_creator.batch_create_graphs(features)
    
    def prepare_data_for_model(self, indices=None, preprocessor=None, 
                               feature_extractor=None, graph_creator=None):
        """
        Prepare data for the GNN-STAN model.
        
        This method runs the entire preprocessing pipeline:
        1. Preprocess fMRI data
        2. Extract features
        3. Create brain graphs
        
        Parameters:
        -----------
        indices : list, optional
            List of subject indices (if None, all subjects are used)
        preprocessor : src.preprocessing.Preprocessor, optional
            Preprocessor instance
        feature_extractor : src.preprocessing.FeatureExtractor, optional
            Feature extractor instance
        graph_creator : src.preprocessing.BrainGraphCreator, optional
            Brain graph creator instance
        
        Returns:
        --------
        tuple
            (graphs, labels) - List of brain graphs and corresponding labels
        """
        if indices is None:
            indices = np.arange(len(self.func_files))
        
        # Step 1: Preprocessing
        print("\n=== Step 1: Preprocessing ===")
        preprocessed_data = self.batch_preprocess(indices, preprocessor)
        
        # Step 2: Feature extraction
        print("\n=== Step 2: Feature Extraction ===")
        if feature_extractor is None:
            from src.preprocessing import FeatureExtractor
            feature_extractor = FeatureExtractor(
                connectivity_type='correlation',
                threshold=0.2,
                dynamic=True,
                window_size=20,
                step_size=5
            )
        
        features = self.extract_features(preprocessed_data, feature_extractor)
        
        # Step 3: Brain graph creation
        print("\n=== Step 3: Brain Graph Creation ===")
        if graph_creator is None:
            from src.preprocessing import BrainGraphCreator
            graph_creator = BrainGraphCreator(
                threshold=0.2,
                use_absolute=True,
                self_loops=False,
                use_dynamic=True
            )
        
        graphs = self.create_brain_graphs(features, graph_creator)
        
        # Extract labels
        labels = np.array([feature['label'] for feature in features])
        
        return graphs, labels