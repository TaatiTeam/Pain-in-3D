"""
Utility functions for loading and filtering datasets based on CSV splits.
"""

import csv
import os
from typing import List, Dict, Set


def load_split_csv(csv_path: str) -> Dict[int, Set[str]]:
    """
    Load split CSV and return mapping of split_id -> set of base annotation names.
    
    Args:
        csv_path: Path to CSV file with columns ['annotation_base_name', 'split']
                  where split is 0=train, 1=val, 2=test
    
    Returns:
        Dictionary mapping split_id to set of annotation base names
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Split CSV file not found: {csv_path}")
    
    splits = {0: set(), 1: set(), 2: set()}  # train, val, test
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            base_name = row['annotation_base_name'].strip()
            split_id = int(row['split'])
            
            if split_id in splits:
                splits[split_id].add(base_name)
            else:
                print(f"Warning: Invalid split_id {split_id} for {base_name}, skipping...")
    
    print(f"Loaded split CSV from {csv_path}:")
    print(f"  Train annotations: {len(splits[0])}")
    print(f"  Val annotations: {len(splits[1])}")
    print(f"  Test annotations: {len(splits[2])}")
    print(f"  Total annotations: {sum(len(s) for s in splits.values())}")
    
    return splits


def filter_data_pairs_by_split(data_pairs: List, target_split: int, splits: Dict[int, Set[str]]) -> List:
    """
    Filter data pairs based on split assignment.
    
    Args:
        data_pairs: List of data pairs from dataset. Each pair should have base_name extractable.
        target_split: Target split ID (0=train, 1=val, 2=test)
        splits: Split mapping from load_split_csv
    
    Returns:
        Filtered list of data pairs for the target split
    """
    if target_split not in splits:
        return []
    
    target_base_names = splits[target_split]
    filtered_pairs = []
    
    for pair in data_pairs:
        # Extract base name from the data pair
        # The structure varies by dataset type, but all should have base_name info
        if len(pair) >= 3:
            # Format: (image_path, annotations, base_name)
            base_name = pair[2]
        else:
            # Format: (image_path, annotations) - extract base_name from image_path
            image_path = pair[0]
            # Extract base name from path, handling different formats
            base_filename = os.path.basename(image_path)
            # Remove extensions and potential suffixes like _heatmap, _view_001, etc.
            base_name = base_filename.split('.')[0]  # Remove extension
            
            # Handle synthetic image suffixes
            for suffix in ['_heatmap', '_view_', '_bg_', '_render_', '_deform_']:
                if suffix in base_name:
                    base_name = base_name.split(suffix)[0]
                    break
        
        if base_name in target_base_names:
            filtered_pairs.append(pair)
    
    print(f"Split {target_split}: Filtered {len(filtered_pairs)} data pairs from {len(data_pairs)} total")
    return filtered_pairs


def extract_base_annotation_name(image_path: str) -> str:
    """
    Extract base annotation name from image path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base annotation name (e.g., 'flame_sample_0000_pain_1')
    """
    base_filename = os.path.basename(image_path)
    
    # Remove extension
    base_name = base_filename.split('.')[0]
    
    # Handle various synthetic image suffixes
    suffixes_to_remove = [
        '_heatmap', '_view_', '_bg_', '_render_', '_deform_', 
        '_image', '_rgb', '_synthetic'
    ]
    
    for suffix in suffixes_to_remove:
        if suffix in base_name:
            base_name = base_name.split(suffix)[0]
            break
    
    return base_name
