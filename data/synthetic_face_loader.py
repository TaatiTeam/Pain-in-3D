import os
import json
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import torch
import pytorch_lightning as pl
from torchvision import transforms
from .split_utils import load_split_csv, filter_data_pairs_by_split

class SyntheticFaceDataset(Dataset):
    def __init__(
        self,
        root_dir='datasets/pain_faces',  # Datasets in root
        transform=None,
        use_neutral_reference=False,
        multi_shot_inference=1,
    ):
        """
        Dataset for synthetic pain face images with Action Unit (AU) annotations for PSPI prediction.
        
        This dataset handles multiple synthetic RGB images (~30) generated per ground truth annotation,
        with different views, backgrounds, etc. It only loads from meshes_inpainted directory.
        
        Args:
            root_dir (string): Directory with pain_faces containing meshes_inpainted/ and annotations/
            transform (callable, optional): Optional transform to be applied on a sample.
            use_neutral_reference (bool): Whether to return a neutral reference image (PSPI=0)
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'meshes_inpainted')  # Only synthetic RGB images
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.transform = transform
        self.use_neutral_reference = use_neutral_reference
        self.multi_shot_inference = int(multi_shot_inference) if multi_shot_inference is not None else 1
        self.subject_to_neutral_images = {}
        
        # Verify meshes_inpainted directory exists
        if not os.path.exists(self.images_dir):
            raise ValueError(f"meshes_inpainted directory not found: {self.images_dir}\n"
                           f"This loader is only for synthetic RGB images, not heatmaps.\n"
                           f"Use heatmap_face_loader.py for heatmap data.")
        
        print(f"Loading synthetic RGB images from: {self.images_dir}")
        
        # Load all annotation files and match them with corresponding synthetic images
        self.data_pairs = self._load_annotations()
        
    def _load_annotations(self):
        """Load all annotation files and match them with corresponding synthetic images (~30 per annotation)."""
        data_pairs = []
        
        # Get all annotation files
        annotation_files = glob.glob(os.path.join(self.annotations_dir, '*.json'))
        
        # OPTIMIZATION: Build image index once instead of globbing per annotation
        # This reduces filesystem operations from O(n*m) to O(1) - HUGE speedup!
        print("Building image index (this may take a moment)...")
        all_images = {}
        # Get all image files once - much faster than globbing per annotation
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in glob.glob(os.path.join(self.images_dir, ext)):
                img_name = os.path.basename(img_path)
                # Extract base name from image filename
                # Handle patterns like: 
                # - base_name_textured_view_000_inpainted.jpg
                # - base_name_view_001.png
                # - base_name_bg_002.jpg
                # - base_name.png (exact match)
                base_name = os.path.splitext(img_name)[0]  # Remove extension first
                
                # Remove common suffixes to get base annotation name
                suffixes_to_remove = [
                    '_textured_view_', '_view_', '_bg_', '_render_', '_deform_', 
                    '_inpainted', '_textured'
                ]
                for suffix in suffixes_to_remove:
                    if suffix in base_name:
                        # Split at suffix and take first part
                        base_name = base_name.split(suffix)[0]
                        break
                
                if base_name not in all_images:
                    all_images[base_name] = []
                all_images[base_name].append(img_path)
        
        total_images_found = 0
        annotations_with_images = 0
        neutral_images_found = 0

        for ann_file in annotation_files:
            # Extract base name from annotation filename
            # e.g., "flame_sample_0000_pain_1_annotations.json" -> "flame_sample_0000_pain_1"
            base_name = os.path.basename(ann_file).replace('_annotations.json', '')
            
            # Load annotation once
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            
            # OPTIMIZATION: Use pre-built index instead of globbing
            matching_images = all_images.get(base_name, [])
            
            if matching_images:
                # Add all matching images with the same annotation
                for image_path in matching_images:
                    data_pairs.append((image_path, annotations, base_name))
                
                total_images_found += len(matching_images)
                annotations_with_images += 1
                
                # Print first few matches for debugging
                if annotations_with_images <= 3:
                    print(f"Annotation: {base_name}")
                    print(f"  Found {len(matching_images)} synthetic images")
                    print(f"  Sample images: {[os.path.basename(img) for img in matching_images[:3]]}")
                    
            else:
                print(f"Warning: No matching synthetic images found for annotation {base_name}")
                print(f"  Searched in: {self.images_dir}")
                print(f"  Pattern: {base_name}_*.png")

        # OPTIMIZATION: Use pre-built index for neutral images
        # Also load neutral images with format flame_sample_XXXX_neutral_*
        # These have all AUs = 0 and PSPI = 0
        print(f"\nLoading neutral images...")
        neutral_images = []
        
        # If using neutral reference, build subject -> neutral images map
        if self.use_neutral_reference:
            self.subject_to_neutral_images = {}
            
        for base_name, img_paths in all_images.items():
            if '_neutral' in base_name:
                neutral_images.extend(img_paths)
                
                if self.use_neutral_reference:
                    # Extract subject ID: flame_sample_XXXX_neutral -> flame_sample_XXXX
                    if '_neutral' in base_name:
                        subject_id = base_name.split('_neutral')[0]
                        if subject_id not in self.subject_to_neutral_images:
                            self.subject_to_neutral_images[subject_id] = []
                        self.subject_to_neutral_images[subject_id].extend(img_paths)
        
        if self.use_neutral_reference:
            print(f"Built neutral reference map for {len(self.subject_to_neutral_images)} synthetic subjects")
        
        # Create neutral annotations (only the AUs used in your pain data = 0, PSPI = 0)
        neutral_annotations = {
            'AU4': 0,   # Inner brow raiser
            'AU6': 0,   # Cheek raiser  
            'AU7': 0,   # Lid tightener
            'AU9': 0,   # Nose wrinkler
            'AU10': 0,  # Upper lip raiser
            'AU43': 0,  # Eyes closed
            'PSPI': 0
        }
        
        # Add neutral images to dataset
        for neutral_image in neutral_images:
            # Extract base name from neutral image filename for consistency
            neutral_base_name = os.path.basename(neutral_image).replace('.png', '').replace('.jpg', '')
            data_pairs.append((neutral_image, neutral_annotations.copy(), neutral_base_name))
            neutral_images_found += 1
            
        if neutral_images_found > 0:
            print(f"  Found {neutral_images_found} neutral images")
            print(f"  Sample neutral images: {[os.path.basename(img) for img in neutral_images[:3]]}")
        else:
            print(f"  No neutral images found with pattern: flame_sample_*_neutral_*")
                
        print(f"\nSynthetic Dataset Summary:")
        print(f"  Total annotations: {len(annotation_files)}")
        print(f"  Annotations with images: {annotations_with_images}")
        print(f"  Total synthetic pain images: {total_images_found}")
        print(f"  Total neutral images: {neutral_images_found}")
        print(f"  Total images: {total_images_found + neutral_images_found}")
        print(f"  Average pain images per annotation: {total_images_found/max(annotations_with_images, 1):.1f}")
        print(f"  Total image-annotation pairs: {len(data_pairs)}")
        
        # Check if we have any data pairs
        if len(data_pairs) == 0:
            raise ValueError(
                f"ERROR: No data pairs found! This usually means:\n"
                f"  1. The meshes_inpainted directory is missing or empty: {self.images_dir}\n"
                f"  2. Image filenames don't match annotation base names\n"
                f"  3. Images are in a different location\n\n"
                f"  Checked {len(annotation_files)} annotation files.\n"
                f"  Images directory: {self.images_dir}\n"
                f"  Directory exists: {os.path.exists(self.images_dir)}\n"
                f"  If directory exists, check if it contains matching image files."
            )
        
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, annotations, base_name = self.data_pairs[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Extract AU values and create feature vector
        au_vector = self._extract_au_features(annotations)
        
        # Calculate PSPI score from AU values (or use provided PSPI)
        pspi_score = self._calculate_pspi(annotations)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            
        sample = {
            'image': image,
            'au_vector': torch.tensor(au_vector, dtype=torch.float32),
            'pspi_score': torch.tensor(pspi_score, dtype=torch.float32),
            'image_path': image_path,
            'base_name': base_name,
        }
        
        if self.use_neutral_reference:
            # Extract subject ID from base_name
            # e.g. flame_sample_XXXX_pain_Y -> flame_sample_XXXX
            # e.g. flame_sample_XXXX_neutral -> flame_sample_XXXX
            subject_id = None
            if '_pain_' in base_name:
                subject_id = base_name.split('_pain_')[0]
            elif '_neutral' in base_name:
                subject_id = base_name.split('_neutral')[0]
            
            neutral_image = None
            if subject_id and subject_id in self.subject_to_neutral_images:
                neutral_paths = self.subject_to_neutral_images[subject_id]
                if neutral_paths:
                    n_shots = max(1, int(self.multi_shot_inference))
                    replace = len(neutral_paths) < n_shots
                    chosen_paths = np.random.choice(neutral_paths, size=n_shots, replace=replace)
                    if n_shots == 1:
                        neutral_path = chosen_paths.item() if hasattr(chosen_paths, "item") else chosen_paths[0]
                        try:
                            neutral_img_pil = Image.open(neutral_path).convert('RGB')
                            neutral_image = self.transform(neutral_img_pil) if self.transform else neutral_img_pil
                        except Exception as e:
                            print(f"Warning: Failed to load neutral image {neutral_path}: {e}")
                    else:
                        neutral_imgs = []
                        for neutral_path in list(chosen_paths):
                            try:
                                neutral_img_pil = Image.open(neutral_path).convert('RGB')
                                neutral_imgs.append(self.transform(neutral_img_pil) if self.transform else neutral_img_pil)
                            except Exception as e:
                                print(f"Warning: Failed to load neutral image {neutral_path}: {e}")
                        if len(neutral_imgs) == n_shots:
                            neutral_image = neutral_imgs
            
            # Fallback
            if neutral_image is None:
                n_shots = max(1, int(self.multi_shot_inference))
                if n_shots > 1:
                    base_neutral = image.clone() if isinstance(image, torch.Tensor) else image.copy()
                    neutral_image = [base_neutral for _ in range(n_shots)]
                else:
                    neutral_image = image.clone() if isinstance(image, torch.Tensor) else image.copy()
                
            sample['neutral_image'] = neutral_image
            
        return sample
    
    def _extract_au_features(self, annotations):
        """Extract AU values as feature vector."""
        # Define the AUs used in PSPI calculation
        aus = ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU43']
        au_vector = []
        
        for au in aus:
            au_vector.append(annotations.get(au, 0))  # Default to 0 if AU not present
            
        return au_vector
    
    def _calculate_pspi(self, annotations):
        """
        Calculate PSPI score from AU values.
        PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
        Range: [0, 16]
        """
        au4 = annotations.get('AU4', 0)
        au6 = annotations.get('AU6', 0)
        au7 = annotations.get('AU7', 0)
        au9 = annotations.get('AU9', 0)
        au10 = annotations.get('AU10', 0)
        au43 = annotations.get('AU43', 0)
        
        pspi = au4 + max(au6, au7) + max(au9, au10) + au43
        
        # Use provided PSPI if available, otherwise use calculated
        provided_pspi = annotations.get('PSPI', pspi)
        
        # Verify calculation matches provided PSPI (for debugging)
        if 'PSPI' in annotations and abs(provided_pspi - pspi) > 0.1:
            print(f"Warning: Calculated PSPI ({pspi}) != Provided PSPI ({provided_pspi})")
            
        return provided_pspi
    
    def get_au_names(self):
        """Return the names of AUs used as features."""
        return ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU43']
    
    def get_pspi_statistics(self):
        """Get statistics about PSPI scores in the dataset."""
        pspi_scores = []
        for _, annotations, _ in self.data_pairs:
            pspi_scores.append(self._calculate_pspi(annotations))
        
        if len(pspi_scores) == 0:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'count': 0
            }
            
        return {
            'min': min(pspi_scores),
            'max': max(pspi_scores),
            'mean': sum(pspi_scores) / len(pspi_scores),
            'count': len(pspi_scores)
        }
    
    def get_unique_annotations_count(self):
        """Get count of unique annotations (ground truth samples)."""
        unique_annotations = set()
        for image_path, _,  _ in self.data_pairs:
            # Extract base annotation name from image path
            basename = os.path.basename(image_path)
            # Remove synthetic image suffixes to get base annotation name
            base_ann = basename.split('_view_')[0].split('_bg_')[0].split('_render_')[0].split('_')[:-1]
            base_ann = '_'.join(base_ann) if isinstance(base_ann, list) else base_ann
            unique_annotations.add(base_ann)
        
        return len(unique_annotations)


class TransformWrapper:
    """Wrapper to apply transforms to dataset samples."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # Also transform neutral image if present
            if 'neutral_image' in sample and sample['neutral_image'] is not None:
                neutral = sample['neutral_image']
                # Multi-shot: list/tuple of PIL images
                if isinstance(neutral, (list, tuple)):
                    transformed = [
                        n if isinstance(n, torch.Tensor) else self.transform(n)
                        for n in neutral
                    ]
                    sample['neutral_image'] = torch.stack(transformed, dim=0)
                else:
                    # Single-shot: only transform if it's a PIL image (avoid double transform)
                    if not isinstance(neutral, torch.Tensor):
                        sample['neutral_image'] = self.transform(neutral)
        return sample


class SyntheticPSPIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="datasets/pain_faces",  # Datasets in root
        batch_size=32,  # Larger batch size for better GPU utilization
        num_workers=8,  # More workers for faster data loading
        image_size=224,
        split_csv=None,             # Path to CSV file with train/val/test splits
        random_seed=42,             # For reproducible operations (kept for compatibility)
        use_entire_dataset=False,   # If True, use entire dataset for testing
        use_neutral_reference=False, # Whether to use a neutral reference image
        multi_shot_inference=1,  # If >1, sample N neutral refs and stack for inference
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_csv = split_csv
        self.random_seed = random_seed
        self.image_size = image_size
        self.use_entire_dataset = use_entire_dataset
        self.use_neutral_reference = use_neutral_reference
        self.multi_shot_inference = int(multi_shot_inference) if multi_shot_inference is not None else 1
        
        # Determine splits based on CSV
        if split_csv:
            print(f"Using CSV-based splits from: {split_csv}")
        elif use_entire_dataset:
            print("Using entire dataset for testing")
        else:
            print("No split CSV provided - using all data for training")
        
        # Define transforms - optimized for speed
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
            transforms.RandomRotation(degrees=5, fill=0),  # Reduced from 30 to 5 degrees for face images (preserves facial geometry)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Re-enabled for overfitting
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random translation and scaling
            # RandomPerspective removed - too aggressive for face images, distorts facial features
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Random blur to simulate focus variations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing."""
        # Load full dataset
        full_dataset = SyntheticFaceDataset(
            root_dir=self.data_dir, 
            transform=None,
            use_neutral_reference=self.use_neutral_reference,
            multi_shot_inference=self.multi_shot_inference,
        )
        
        if self.use_entire_dataset:
            # Use all data for testing
            train_dataset = SyntheticFaceDataset(
                root_dir=self.data_dir,
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            train_dataset.data_pairs = []
            val_dataset = SyntheticFaceDataset(
                root_dir=self.data_dir,
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            val_dataset.data_pairs = []
            test_dataset = full_dataset
            
            print(f"Using entire dataset for testing:")
            print(f"  Train: 0 samples")
            print(f"  Val:   0 samples")
            print(f"  Test:  {len(test_dataset)} samples")
            
        # Handle CSV-based splitting
        elif self.split_csv:
            # Load splits from CSV
            splits = load_split_csv(self.split_csv)
            
            # Filter data pairs by split
            train_pairs = filter_data_pairs_by_split(full_dataset.data_pairs, 0, splits)  # 0 = train
            val_pairs = filter_data_pairs_by_split(full_dataset.data_pairs, 1, splits)    # 1 = val
            test_pairs = filter_data_pairs_by_split(full_dataset.data_pairs, 2, splits)   # 2 = test
            
            # OPTIMIZATION: Create lightweight dataset wrappers instead of full reload
            # This avoids calling __init__ and _load_annotations() again (saves ~3-4 hours!)
            class DatasetWrapper(SyntheticFaceDataset):
                """Lightweight wrapper that skips __init__ and uses provided data_pairs."""
                def __init__(self, data_pairs, root_dir, transform=None, use_neutral_reference=False, subject_to_neutral_images=None):
                    # Skip the parent __init__ that loads all data
                    # Just set the minimal attributes needed
                    self.root_dir = root_dir
                    self.images_dir = os.path.join(root_dir, 'meshes_inpainted')
                    self.annotations_dir = os.path.join(root_dir, 'annotations')
                    self.transform = transform
                    self.data_pairs = data_pairs
                    self.use_neutral_reference = use_neutral_reference
                    self.multi_shot_inference = int(getattr(full_dataset, 'multi_shot_inference', 1))
                    self.subject_to_neutral_images = subject_to_neutral_images or {}
                
                # Inherit all other methods from SyntheticFaceDataset
                # __getitem__, __len__, _extract_au_features, _calculate_pspi, etc.
            
            train_dataset = DatasetWrapper(
                train_pairs, 
                self.data_dir, 
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                subject_to_neutral_images=full_dataset.subject_to_neutral_images
            )
            val_dataset = DatasetWrapper(
                val_pairs, 
                self.data_dir, 
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                subject_to_neutral_images=full_dataset.subject_to_neutral_images
            )
            test_dataset = DatasetWrapper(
                test_pairs, 
                self.data_dir, 
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                subject_to_neutral_images=full_dataset.subject_to_neutral_images
            )
            
            print(f"CSV-based splits:")
            print(f"  Train: {len(train_pairs)} samples")
            print(f"  Val:   {len(val_pairs)} samples")
            print(f"  Test:  {len(test_pairs)} samples")
            print(f"  Total: {len(full_dataset.data_pairs)} samples")
            
        else:
            # Use all data for training (no validation/test)
            train_dataset = full_dataset
            val_dataset = SyntheticFaceDataset(
                root_dir=self.data_dir,
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            val_dataset.data_pairs = []  # Empty dataset
            test_dataset = SyntheticFaceDataset(
                root_dir=self.data_dir,
                transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            test_dataset.data_pairs = []  # Empty dataset
            
            print(f"No CSV split provided - using all data for training:")
            print(f"  Train: {len(train_dataset.data_pairs)} samples")
            print(f"  Val:   0 samples")
            print(f"  Test:  0 samples")
        
        # Apply transforms by wrapping datasets
        self.train_dataset = TransformWrapper(train_dataset, self.train_transform)
        self.val_dataset = TransformWrapper(val_dataset, self.val_test_transform)
        self.test_dataset = TransformWrapper(test_dataset, self.val_test_transform)
        
        # Print dataset statistics
        print(f"\nSynthetic Dataset Statistics:")
        print(f"Random seed used: {self.random_seed}")
        if hasattr(full_dataset, 'get_pspi_statistics'):
            stats = full_dataset.get_pspi_statistics()
            print(f"PSPI range: [{stats['min']:.1f}, {stats['max']:.1f}]")
            print(f"PSPI mean: {stats['mean']:.2f}")
            print(f"AU features: {full_dataset.get_au_names()}")
            print(f"Total synthetic images: {len(full_dataset)}")
            print(f"Unique ground truth annotations: {full_dataset.get_unique_annotations_count()}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,  # Prefetch batches per worker
            drop_last=True  # Drop incomplete batches for consistent GPU utilization
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4,  # Prefetch 4 batches per worker
            drop_last=False  # Keep all validation samples
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
