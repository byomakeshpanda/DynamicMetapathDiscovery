from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
def create_node_splits(data, test_size=0.1, val_size=0.1, random_state=42):
    """Create proper train/val/test splits for author nodes"""
    author_indices = np.arange(data['author'].num_nodes)
    y = data['author'].y.numpy()
    
    # First split: train+val (90%) vs test (10%)
    train_val_idx, test_idx = train_test_split(
        author_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )
    
    # Second split: train (80%) vs val (10% of total)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size/(1-test_size),
        random_state=random_state,
        stratify=y[train_val_idx]
    )
    
    # Create masks
    data['author'].train_mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
    data['author'].val_mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
    data['author'].test_mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
    
    data['author'].train_mask[train_idx] = True
    data['author'].val_mask[val_idx] = True
    data['author'].test_mask[test_idx] = True
    
    print(f"Final split sizes:")
    print(f"Training: {len(train_idx)} ({len(train_idx)/data['author'].num_nodes:.1%})")
    print(f"Validation: {len(val_idx)} ({len(val_idx)/data['author'].num_nodes:.1%})")
    print(f"Test: {len(test_idx)} ({len(test_idx)/data['author'].num_nodes:.1%})")
    
    return data
def load_dblp_dataset(data_root: str = '/data') -> HeteroData:
    """Load DBLP dataset with proper existence checks"""
    # PyG's DBLP dataset automatically checks for existing files
    dataset = DBLP(root=data_root)
    data = dataset[0]

    # Verify we're using existing data
    if not os.path.exists(os.path.join(data_root, 'processed')):
        raise FileNotFoundError("Dataset not downloaded properly")

    return data
def get_dblp_data(data_root: str = '/data') -> HeteroData:
    """Updated data loading pipeline with proper node splits"""
    data = load_dblp_dataset(data_root)
    data = create_node_splits(data)  # Use node splits instead of edge splits
    return data