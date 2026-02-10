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

class Pain3DDataset(Dataset):
    def __init__(
        self,
        root_dir='datasets/pain_faces',
        transform=None,
        use_neutral_reference=False,
        multi_shot_inference=1,
    ):
        """
        Dataset for synthetic pain face images with AU annotations for PSPI prediction.

        Handles multiple synthetic RGB images (~30) per ground truth annotation,
        with different views, backgrounds, etc. Only loads from meshes_inpainted directory.

        Args:
            root_dir: Directory with pain_faces containing meshes_inpainted/ and annotations/
            transform: Optional transform to be applied on a sample.
            use_neutral_reference: Whether to return a neutral reference image (PSPI=0)
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'meshes_inpainted')  # Synthetic RGB images only
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.transform = transform
        self.use_neutral_reference = use_neutral_reference
        self.multi_shot_inference = int(multi_shot_inference) if multi_shot_inference is not None else 1
        self.subject_to_neutral_images = {}

        if not os.path.exists(self.images_dir):
            raise ValueError(f"meshes_inpainted directory not found: {self.images_dir}\n"
                           f"This loader is only for synthetic RGB images, not heatmaps.\n"
                           f"Use heatmap_face_loader.py for heatmap data.")

        self.data_pairs = self._load_annotations()

    def _load_annotations(self):
        """Load all annotation files and match with corresponding synthetic images (~30 per annotation)."""
        data_pairs = []
        annotation_files = glob.glob(os.path.join(self.annotations_dir, '*.json'))

        # Build image index once to avoid per-annotation globbing
        all_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in glob.glob(os.path.join(self.images_dir, ext)):
                img_name = os.path.basename(img_path)
                # Handle patterns like:
                # - base_name_textured_view_000_inpainted.jpg
                # - base_name_view_001.png
                # - base_name_bg_002.jpg
                # - base_name.png (exact match)
                base_name = os.path.splitext(img_name)[0]

                suffixes_to_remove = [
                    '_textured_view_', '_view_', '_bg_', '_render_', '_deform_',
                    '_inpainted', '_textured'
                ]
                for suffix in suffixes_to_remove:
                    if suffix in base_name:
                        base_name = base_name.split(suffix)[0]
                        break

                if base_name not in all_images:
                    all_images[base_name] = []
                all_images[base_name].append(img_path)

        for ann_file in annotation_files:
            base_name = os.path.basename(ann_file).replace('_annotations.json', '')

            with open(ann_file, 'r') as f:
                annotations = json.load(f)

            matching_images = all_images.get(base_name, [])

            if matching_images:
                for image_path in matching_images:
                    data_pairs.append((image_path, annotations, base_name))

        # Also load neutral images (all AUs = 0, PSPI = 0)
        neutral_images = []

        if self.use_neutral_reference:
            self.subject_to_neutral_images = {}

        for base_name, img_paths in all_images.items():
            if '_neutral' in base_name:
                neutral_images.extend(img_paths)

                if self.use_neutral_reference:
                    # e.g. flame_sample_XXXX_neutral -> flame_sample_XXXX
                    if '_neutral' in base_name:
                        subject_id = base_name.split('_neutral')[0]
                        if subject_id not in self.subject_to_neutral_images:
                            self.subject_to_neutral_images[subject_id] = []
                        self.subject_to_neutral_images[subject_id].extend(img_paths)

        neutral_annotations = {
            'AU4': 0,   # Inner brow raiser
            'AU6': 0,   # Cheek raiser
            'AU7': 0,   # Lid tightener
            'AU9': 0,   # Nose wrinkler
            'AU10': 0,  # Upper lip raiser
            'AU43': 0,  # Eyes closed
            'PSPI': 0
        }

        for neutral_image in neutral_images:
            neutral_base_name = os.path.basename(neutral_image).replace('.png', '').replace('.jpg', '')
            data_pairs.append((neutral_image, neutral_annotations.copy(), neutral_base_name))

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
        image = Image.open(image_path).convert('RGB')
        au_vector = self._extract_au_features(annotations)
        pspi_score = self._calculate_pspi(annotations)

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
                        except Exception:
                            pass
                    else:
                        neutral_imgs = []
                        for neutral_path in list(chosen_paths):
                            try:
                                neutral_img_pil = Image.open(neutral_path).convert('RGB')
                                neutral_imgs.append(self.transform(neutral_img_pil) if self.transform else neutral_img_pil)
                            except Exception:
                                pass
                        if len(neutral_imgs) == n_shots:
                            neutral_image = neutral_imgs

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
        aus = ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU43']
        return [annotations.get(au, 0) for au in aus]

    def _calculate_pspi(self, annotations):
        """
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
        return annotations.get('PSPI', pspi)

    def get_au_names(self):
        return ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU43']

    def get_pspi_statistics(self):
        pspi_scores = []
        for _, annotations, _ in self.data_pairs:
            pspi_scores.append(self._calculate_pspi(annotations))

        if len(pspi_scores) == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'count': 0}

        return {
            'min': min(pspi_scores),
            'max': max(pspi_scores),
            'mean': sum(pspi_scores) / len(pspi_scores),
            'count': len(pspi_scores)
        }

    def get_unique_annotations_count(self):
        unique_annotations = set()
        for image_path, _, _ in self.data_pairs:
            basename = os.path.basename(image_path)
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
            if 'neutral_image' in sample and sample['neutral_image'] is not None:
                neutral = sample['neutral_image']
                if isinstance(neutral, (list, tuple)):
                    transformed = [
                        n if isinstance(n, torch.Tensor) else self.transform(n)
                        for n in neutral
                    ]
                    sample['neutral_image'] = torch.stack(transformed, dim=0)
                else:
                    # Only transform PIL images (avoid double transform on tensors)
                    if not isinstance(neutral, torch.Tensor):
                        sample['neutral_image'] = self.transform(neutral)
        return sample


class Pain3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="datasets/pain_faces",
        batch_size=32,
        num_workers=8,
        image_size=224,
        split_csv=None,
        random_seed=42,
        use_entire_dataset=False,
        use_neutral_reference=False,
        multi_shot_inference=1,
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

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5, fill=0),  # Kept small to preserve facial geometry
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # No RandomPerspective -- too aggressive for faces, distorts features
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        full_dataset = Pain3DDataset(
            root_dir=self.data_dir,
            transform=None,
            use_neutral_reference=self.use_neutral_reference,
            multi_shot_inference=self.multi_shot_inference,
        )

        if self.use_entire_dataset:
            train_dataset = Pain3DDataset(
                root_dir=self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            train_dataset.data_pairs = []
            val_dataset = Pain3DDataset(
                root_dir=self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            val_dataset.data_pairs = []
            test_dataset = full_dataset

        elif self.split_csv:
            splits = load_split_csv(self.split_csv)
            train_pairs = filter_data_pairs_by_split(full_dataset.data_pairs, 0, splits)  # 0 = train
            val_pairs = filter_data_pairs_by_split(full_dataset.data_pairs, 1, splits)    # 1 = val
            test_pairs = filter_data_pairs_by_split(full_dataset.data_pairs, 2, splits)   # 2 = test

            # Lightweight wrapper to avoid re-running _load_annotations() (saves ~3-4 hours)
            class DatasetWrapper(Pain3DDataset):
                """Skips __init__ and uses pre-filtered data_pairs."""
                def __init__(self, data_pairs, root_dir, transform=None, use_neutral_reference=False, subject_to_neutral_images=None):
                    self.root_dir = root_dir
                    self.images_dir = os.path.join(root_dir, 'meshes_inpainted')
                    self.annotations_dir = os.path.join(root_dir, 'annotations')
                    self.transform = transform
                    self.data_pairs = data_pairs
                    self.use_neutral_reference = use_neutral_reference
                    self.multi_shot_inference = int(getattr(full_dataset, 'multi_shot_inference', 1))
                    self.subject_to_neutral_images = subject_to_neutral_images or {}

            train_dataset = DatasetWrapper(
                train_pairs, self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                subject_to_neutral_images=full_dataset.subject_to_neutral_images
            )
            val_dataset = DatasetWrapper(
                val_pairs, self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                subject_to_neutral_images=full_dataset.subject_to_neutral_images
            )
            test_dataset = DatasetWrapper(
                test_pairs, self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                subject_to_neutral_images=full_dataset.subject_to_neutral_images
            )

        else:
            # No split CSV: use all data for training
            train_dataset = full_dataset
            val_dataset = Pain3DDataset(
                root_dir=self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            val_dataset.data_pairs = []
            test_dataset = Pain3DDataset(
                root_dir=self.data_dir, transform=None,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )
            test_dataset.data_pairs = []

        self.train_dataset = TransformWrapper(train_dataset, self.train_transform)
        self.val_dataset = TransformWrapper(val_dataset, self.val_test_transform)
        self.test_dataset = TransformWrapper(test_dataset, self.val_test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4,
            drop_last=False
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
