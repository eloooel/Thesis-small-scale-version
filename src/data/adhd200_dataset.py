"""
Data loader for the ADHD-200 dataset.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.data import Data, Dataset
import torch


class ADHD200Dataset:
    """
    Class to handle loading and preprocessing of the ADHD-200 dataset.
    
    This class provides functionality to:
    1. Download the dataset (or use local copy)
    2. Extract phenotypic information
    3. Filter subjects based on criteria
    4. Split data into train/test sets
    5. Create cross-validation folds
    6. Generate PyTorch Geometric Data objects
    """
    
    def __init__(self, data_dir='data/adhd200', n_subjects=None, download=True, 
                 atlas='schaefer_200', test_size=0.2, random_state=42):
        """
        Initialize the ADHD-200 dataset.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory to store the dataset
        n_subjects : int, optional
            Number of subjects to load (None for all)
        download : bool, optional
            Whether to download the dataset if not found locally
        atlas : str, optional
            Atlas to use for parcellation ('schaefer_200', 'aal116', etc.)
        test_size : float, optional
            Fraction of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.n_subjects = n_subjects
        self.download = download
        self.atlas = atlas
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize empty attributes
        self.adhd_dataset = None
        self.phenotypic = None
        self.func_files = None
        self.confounds = None
        self.labels = None
        self.sites = None
        self.ages = None
        self.genders = None
        self.train_indices = None
        self.test_indices = None
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Load the ADHD-200 dataset.
        
        If download=True and the dataset is not found locally, it will be downloaded.
        """
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if dataset exists locally
        local_dataset_path = os.path.join(self.data_dir, 'adhd200_metadata.csv')
        
        if not os.path.exists(local_dataset_path) or self.download:
            print(f"Downloading ADHD-200 dataset (n_subjects={self.n_subjects})...")
            self.adhd_dataset = datasets.fetch_adhd(n_subjects=self.n_subjects, data_dir=self.data_dir)
            
            # Save phenotypic data for future use
            self.phenotypic = self.adhd_dataset.phenotypic
            self.phenotypic.to_csv(local_dataset_path, index=False)
        else:
            print("Loading ADHD-200 dataset from local files...")
            self.phenotypic = pd.read_csv(local_dataset_path)
            
            # Load functional files paths
            func_files_path = os.path.join(self.data_dir, 'func_files.npy')
            if os.path.exists(func_files_path):
                self.func_files = np.load(func_files_path, allow_pickle=True)
            else:
                # This will need to be adjusted based on the actual file structure
                print("WARNING: Functional files not found locally. Please download the dataset.")
        
        # Extract information
        if self.adhd_dataset is not None:
            self.func_files = self.adhd_dataset.func
        
        self.labels = self.phenotypic['adhd'].values  # 1 for ADHD, 0 for control
        self.sites = self.phenotypic['site'].values
        
        # Extract age and sex if available
        if 'age' in self.phenotypic.columns:
            self.ages = self.phenotypic['age'].values
        
        if 'gender' in self.phenotypic.columns:
            self.genders = self.phenotypic['gender'].values
        
        print(f"Loaded dataset with {len(self.labels)} subjects ({sum(self.labels)} ADHD, {len(self.labels) - sum(self.labels)} control)")
        print(f"Sites: {np.unique(self.sites)}")
    
    def create_train_test_split(self, stratify_by='site'):
        """
        Create train/test split.
        
        Parameters:
        -----------
        stratify_by : str, optional
            How to stratify the split ('label', 'site', or 'both')
        
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
        elif stratify_by == 'both':
            # Create a combined stratification variable
            stratify = [f"{site}_{label}" for site, label in zip(self.sites, self.labels)]
        else:
            stratify = None
        
        self.train_indices, self.test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        print(f"Created train/test split: {len(self.train_indices)} train, {len(self.test_indices)} test")
        
        return self.train_indices, self.test_indices
    
    def create_cv_folds(self, n_folds=5, stratify_by='site'):
        """
        Create cross-validation folds.
        
        Parameters:
        -----------
        n_folds : int, optional
            Number of folds
        stratify_by : str, optional
            How to stratify the folds ('label', 'site', or 'both')
        
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
        elif stratify_by == 'both':
            # Create a combined stratification variable
            stratify = [f"{site}_{label}" for site, label in zip(self.sites, self.labels)]
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
            val_indices = np.where(self.sites == site)[0]
            
            # Train indices are subjects from all other sites
            train_indices = np.where(self.sites != site)[0]
            
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
        subject_id = f"subject_{index}"
        
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
            
            # Add age and gender if available
            if hasattr(self, 'ages') and self.ages is not None:
                result['age'] = self.ages[index]
                
            if hasattr(self, 'genders') and self.genders is not None:
                result['gender'] = self.genders[index]
            
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
            indices = np.arange(len(self.labels))
        
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