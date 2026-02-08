"""
UNBC-McMaster Shoulder Pain Expression Archive Dataset Loader.

Loads UNBC data from HDF5 files, supports PSPI regression and binary classification,
leave-one-subject-out and 5-fold cross-validation, AU multi-task learning,
and optional neutral reference images for cross-attention.
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
import os
import math
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class UNBCDataset(Dataset):
    """UNBC-McMaster Shoulder Pain Expression Archive Dataset."""

    def __init__(
        self,
        data_dir="datasets/UNBC-McMaster",
        transform=None,
        mode="train",
        fold=None,
        cv_protocol="5fold",
        return_aus=True,
        image_size=160,
        use_neutral_reference=False,
        multi_shot_inference=1,
    ):
        """
        Args:
            data_dir: Directory containing UNBC HDF5 files
            transform: Optional image transform
            mode: 'train', 'val', or 'test'
            fold: CV fold index (0-4 for 5fold, 0-24 for LOSO)
            cv_protocol: '5fold' or 'loso' (leave-one-subject-out)
            return_aus: Whether to return AU features
            image_size: Target image size
            use_neutral_reference: Whether to return a neutral reference image (PSPI=0)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.fold = fold
        self.cv_protocol = cv_protocol
        self.return_aus = return_aus
        self.image_size = image_size
        self.use_neutral_reference = use_neutral_reference
        self.multi_shot_inference = int(multi_shot_inference) if multi_shot_inference is not None else 1

        self.frames_file = os.path.join(data_dir, "frames_unbc_2020-09-21-05-42-04.hdf5")
        self.annotations_file = os.path.join(data_dir, "annotations_unbc_2020-10-13-22-55-04.hdf5")
        self.folds_file = os.path.join(data_dir, "UNBC_CVFolds_2019-05-16-15-16-36.hdf5")

        for file_path in [self.frames_file, self.annotations_file, self.folds_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        self._load_data()

    def _load_data(self):
        """Load and index the dataset using subject-based splits."""

        with h5py.File(self.frames_file, 'r') as frames_f, \
             h5py.File(self.annotations_file, 'r') as ann_f, \
             h5py.File(self.folds_file, 'r') as folds_f:

            self.images = frames_f['image']
            self.indexes = frames_f['index'][:]
            self.subject_names = ann_f['subject_name'][:]

            # Filter out annotations pointing to non-existent frames
            max_frame_idx = len(self.images) - 1
            ann_indexes = ann_f['index'][:]
            valid_ann_mask = ann_indexes <= max_frame_idx

            if not np.all(valid_ann_mask):
                self.subject_names = self.subject_names[valid_ann_mask]
                ann_indexes = ann_indexes[valid_ann_mask]

            au_keys = [key for key in ann_f.keys() if key.startswith('label_AU_')]

            if not np.all(valid_ann_mask):
                self.au4 = ann_f['label_AU_4'][:][valid_ann_mask] if 'label_AU_4' in ann_f else np.zeros(len(self.subject_names))
                self.au6 = ann_f['label_AU_6'][:][valid_ann_mask] if 'label_AU_6' in ann_f else np.zeros(len(self.subject_names))
                self.au7 = ann_f['label_AU_7'][:][valid_ann_mask] if 'label_AU_7' in ann_f else np.zeros(len(self.subject_names))
                self.au9 = ann_f['label_AU_9'][:][valid_ann_mask] if 'label_AU_9' in ann_f else np.zeros(len(self.subject_names))
                self.au10 = ann_f['label_AU_10'][:][valid_ann_mask] if 'label_AU_10' in ann_f else np.zeros(len(self.subject_names))
                self.au43 = ann_f['label_AU_43'][:][valid_ann_mask] if 'label_AU_43' in ann_f else np.zeros(len(self.subject_names))

                if 'label_pspi' in ann_f:
                    self.pspi_scores = ann_f['label_pspi'][:][valid_ann_mask]
                else:
                    # PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
                    self.pspi_scores = self._calculate_pspi_from_aus()
            else:
                self.au4 = ann_f['label_AU_4'][:] if 'label_AU_4' in ann_f else np.zeros(len(self.indexes))
                self.au6 = ann_f['label_AU_6'][:] if 'label_AU_6' in ann_f else np.zeros(len(self.indexes))
                self.au7 = ann_f['label_AU_7'][:] if 'label_AU_7' in ann_f else np.zeros(len(self.indexes))
                self.au9 = ann_f['label_AU_9'][:] if 'label_AU_9' in ann_f else np.zeros(len(self.indexes))
                self.au10 = ann_f['label_AU_10'][:] if 'label_AU_10' in ann_f else np.zeros(len(self.indexes))
                self.au43 = ann_f['label_AU_43'][:] if 'label_AU_43' in ann_f else np.zeros(len(self.indexes))

                # Load PSPI if available, otherwise calculate from AUs
                if 'label_pspi' in ann_f:
                    self.pspi_scores = ann_f['label_pspi'][:]
                else:
                    self.pspi_scores = self._calculate_pspi_from_aus()

            self.au4 = np.nan_to_num(self.au4, nan=0.0)
            self.au6 = np.nan_to_num(self.au6, nan=0.0)
            self.au7 = np.nan_to_num(self.au7, nan=0.0)
            self.au9 = np.nan_to_num(self.au9, nan=0.0)
            self.au10 = np.nan_to_num(self.au10, nan=0.0)
            self.au43 = np.nan_to_num(self.au43, nan=0.0)
            self.pspi_scores = np.nan_to_num(self.pspi_scores, nan=0.0)

            if self.use_neutral_reference:
                self.subject_to_neutral_indices = {}

                all_subject_numbers = []
                for name in self.subject_names:
                    name_str = name.decode() if isinstance(name, bytes) else name
                    number = int(name_str.split('-')[0])
                    all_subject_numbers.append(number)
                all_subject_numbers = np.array(all_subject_numbers)

                # Neutral frames are those with PSPI == 0
                for idx, pspi in enumerate(self.pspi_scores):
                    if pspi == 0:
                        subj_num = all_subject_numbers[idx]
                        if subj_num not in self.subject_to_neutral_indices:
                            self.subject_to_neutral_indices[subj_num] = []
                        self.subject_to_neutral_indices[subj_num].append(idx)

            subject_numbers = []
            for name in self.subject_names:
                name_str = name.decode() if isinstance(name, bytes) else name
                number = int(name_str.split('-')[0])
                subject_numbers.append(number)
            subject_numbers = np.array(subject_numbers)

            if self.fold is not None:
                if self.cv_protocol not in {"5fold", "loso"}:
                    raise ValueError(f"Invalid cv_protocol={self.cv_protocol!r}. Expected '5fold' or 'loso'.")

                if self.cv_protocol == "5fold":
                    assert 0 <= int(self.fold) < 5, f"Fold index must be 0-4 for cv_protocol='5fold', got {self.fold}"

                    fold_key = f'fold-{int(self.fold)}'
                    if fold_key not in folds_f:
                        raise KeyError(f"Fold {self.fold} not found in CV folds file")

                    val_subjects = set(folds_f[fold_key]['validation_subjects'][:])
                    train_subjects = set(folds_f[fold_key]['train_subjects'][:])
                else:
                    # LOSO: each fold holds out one subject (typically 0-24)
                    subject_ids = sorted(list(set(subject_numbers.tolist())))
                    n_subjects = len(subject_ids)
                    if n_subjects == 0:
                        raise RuntimeError("No subjects found in UNBC annotations; cannot build LOSO splits.")

                    fold_idx = int(self.fold)
                    if not (0 <= fold_idx < n_subjects):
                        raise ValueError(
                            f"Fold index must be 0-{n_subjects - 1} for cv_protocol='loso', got {self.fold}. "
                            f"(Detected {n_subjects} unique subjects: {subject_ids})"
                        )

                    held_out_subject = int(subject_ids[fold_idx])
                    val_subjects = {held_out_subject}
                    train_subjects = set(subject_ids) - {held_out_subject}

                self.valid_indices = []
                if self.mode == 'train':
                    target_subjects = train_subjects
                elif self.mode in ('val', 'test'):
                    target_subjects = val_subjects
                else:
                    target_subjects = train_subjects | val_subjects

                for subject_num in target_subjects:
                    subject_indices = np.where(subject_numbers == subject_num)[0]
                    self.valid_indices.extend(subject_indices)
                self.valid_indices = np.array(sorted(self.valid_indices))

            else:
                self.valid_indices = np.arange(len(self.indexes))

    def _calculate_pspi_from_aus(self):
        """Calculate PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43."""
        def safe_max(*arrays):
            stacked = np.stack(arrays, axis=-1)
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore", category=RuntimeWarning)
                result = np.nanmax(stacked, axis=-1)
            return np.nan_to_num(result, nan=0.0)

        def safe_val(array):
            return np.nan_to_num(array, nan=0.0)

        def safe_nanmax(val):
            if hasattr(val, '__iter__'):
                if np.all(np.isnan(val)):
                    return 0.0
                with np.warnings.catch_warnings():
                    np.warnings.simplefilter("ignore", category=RuntimeWarning)
                    result = np.nanmax(val)
                return safe_val(result)
            else:
                return safe_val(val)

        au4_max = np.array([safe_nanmax(self.au4[idx]) for idx in range(len(self.au4))])
        au6_max = np.array([safe_nanmax(self.au6[idx]) for idx in range(len(self.au6))])
        au7_max = np.array([safe_nanmax(self.au7[idx]) for idx in range(len(self.au7))])
        au9_max = np.array([safe_nanmax(self.au9[idx]) for idx in range(len(self.au9))])
        au10_max = np.array([safe_nanmax(self.au10[idx]) for idx in range(len(self.au10))])
        au43_max = np.array([safe_nanmax(self.au43[idx]) for idx in range(len(self.au43))])

        pspi = (au4_max +
                safe_max(au6_max, au7_max) +
                safe_max(au9_max, au10_max) +
                au43_max)

        return pspi

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        with h5py.File(self.frames_file, 'r') as f:
            image_data = f['image'][actual_idx]  # (H, W, C)
            if len(image_data.shape) == 3:
                image = Image.fromarray(image_data.astype(np.uint8))
            else:
                image = Image.fromarray(image_data.astype(np.uint8)).convert('RGB')

        # Keep untransformed copy for neutral-reference fallback
        image_pil = image
        pspi_score = float(self.pspi_scores[actual_idx])

        au_features = None
        if self.return_aus:
            def get_au_val(au_array, idx):
                """Get AU value, handling NaN and returning continuous float."""
                val = au_array[idx]
                if hasattr(val, '__iter__'):
                    if np.all(np.isnan(val)):
                        return 0.0
                    max_val = np.nanmax(val)
                    if np.isnan(max_val):
                        return 0.0
                    return float(max_val)
                else:
                    return 0.0 if np.isnan(val) else float(val)

            # Return continuous AU values (not discrete classes), following reference implementation
            au4_val = get_au_val(self.au4, actual_idx)
            au6_val = get_au_val(self.au6, actual_idx)
            au7_val = get_au_val(self.au7, actual_idx)
            au9_val = get_au_val(self.au9, actual_idx)
            au10_val = get_au_val(self.au10, actual_idx)
            au43_val = get_au_val(self.au43, actual_idx)
            au43_bin = 1.0 if au43_val > 0 else 0.0  # AU43 is binarized

            au_features = np.array([
                au4_val, au6_val, au7_val, au9_val, au10_val, au43_bin
            ], dtype=np.float32)

        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)

        sample = {
            'image': image,
            'pspi_score': torch.tensor(pspi_score, dtype=torch.float32),
            'image_path': f"unbc_frame_{actual_idx}",
            'base_name': f"unbc_{actual_idx}",
        }

        if au_features is not None:
            sample['au_vector'] = torch.tensor(au_features, dtype=torch.float32)

        if self.use_neutral_reference:
            name = self.subject_names[actual_idx]
            name_str = name.decode() if isinstance(name, bytes) else name
            subject_num = int(name_str.split('-')[0])

            neutral_indices = self.subject_to_neutral_indices.get(subject_num, [])

            n_shots = max(1, int(self.multi_shot_inference))
            neutral_pils = []
            if neutral_indices:
                replace = len(neutral_indices) < n_shots
                chosen = np.random.choice(neutral_indices, size=n_shots, replace=replace)
                with h5py.File(self.frames_file, 'r') as f:
                    for neutral_idx in list(chosen):
                        neutral_image_data = f['image'][neutral_idx]
                        if len(neutral_image_data.shape) == 3:
                            neutral_pils.append(Image.fromarray(neutral_image_data.astype(np.uint8)))
                        else:
                            neutral_pils.append(Image.fromarray(neutral_image_data.astype(np.uint8)).convert('RGB'))

            if not neutral_pils:
                # Fallback: use current image as neutral
                neutral_pils = [image_pil.copy() for _ in range(n_shots)]

            if self.transform:
                neutral_tensors = [self.transform(im) for im in neutral_pils]
            else:
                transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                neutral_tensors = [transform(im) for im in neutral_pils]

            if n_shots == 1:
                sample['neutral_image'] = neutral_tensors[0]
            else:
                sample['neutral_image'] = torch.stack(neutral_tensors, dim=0)

        return sample

    def get_au_names(self):
        return ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU43']

    def get_subject_info(self):
        if not hasattr(self, 'subject_names') or not hasattr(self, 'valid_indices'):
            return {"message": "Subject information not available"}

        subject_numbers = []
        for idx in self.valid_indices:
            name = self.subject_names[idx]
            name_str = name.decode() if isinstance(name, bytes) else name
            number = int(name_str.split('-')[0])
            subject_numbers.append(number)

        unique_subjects = sorted(list(set(subject_numbers)))
        subject_counts = {}
        for subj in unique_subjects:
            subject_counts[subj] = subject_numbers.count(subj)

        return {
            'total_subjects': len(unique_subjects),
            'subject_ids': unique_subjects,
            'subject_frame_counts': subject_counts,
            'total_frames': len(self.valid_indices)
        }

    def get_pspi_statistics(self):
        valid_scores = self.pspi_scores[self.valid_indices]
        return {
            'min': float(valid_scores.min()),
            'max': float(valid_scores.max()),
            'mean': float(valid_scores.mean()),
            'std': float(valid_scores.std()),
            'count': len(valid_scores)
        }


class UNBCDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for UNBC-McMaster dataset."""

    def __init__(
        self,
        data_dir="datasets/UNBC-McMaster",
        batch_size=32,
        num_workers=4,
        image_size=160,
        fold=0,
        cv_protocol="5fold",
        return_aus=True,
        pin_memory=True,
        use_neutral_reference=False,
        multi_shot_inference=1,
        use_weighted_sampling=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.fold = fold
        self.cv_protocol = cv_protocol
        self.return_aus = return_aus
        self.pin_memory = pin_memory
        self.use_neutral_reference = use_neutral_reference
        self.multi_shot_inference = int(multi_shot_inference) if multi_shot_inference is not None else 1
        self.use_weighted_sampling = use_weighted_sampling

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = UNBCDataset(
                data_dir=self.data_dir,
                transform=self.train_transform,
                mode="train",
                fold=self.fold,
                cv_protocol=self.cv_protocol,
                return_aus=self.return_aus,
                image_size=self.image_size,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )

            self.val_dataset = UNBCDataset(
                data_dir=self.data_dir,
                transform=self.val_test_transform,
                mode="val",
                fold=self.fold,
                cv_protocol=self.cv_protocol,
                return_aus=self.return_aus,
                image_size=self.image_size,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )

        if stage == "test" or stage is None:
            self.test_dataset = UNBCDataset(
                data_dir=self.data_dir,
                transform=self.val_test_transform,
                mode="test",
                fold=self.fold,
                cv_protocol=self.cv_protocol,
                return_aus=self.return_aus,
                image_size=self.image_size,
                use_neutral_reference=self.use_neutral_reference,
                multi_shot_inference=self.multi_shot_inference,
            )

    def train_dataloader(self):
        if not self.use_weighted_sampling:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )

        dataset = self.train_dataset
        all_targets = []

        def extract_targets(ds):
            if hasattr(ds, 'pspi_scores') and hasattr(ds, 'valid_indices'):
                return ds.pspi_scores[ds.valid_indices]  # UNBCDataset
            elif hasattr(ds, 'data_pairs'):
                return [pair[1].get('pspi', 0) for pair in ds.data_pairs]  # Pain3DDataset
            elif hasattr(ds, 'dataset'):
                return extract_targets(ds.dataset)  # TransformWrapper
            return None

        if isinstance(dataset, torch.utils.data.ConcatDataset):
            for ds in dataset.datasets:
                targets = extract_targets(ds)
                if targets is not None:
                    all_targets.extend(targets)
                else:
                    # Can't extract targets; fall back to random shuffle
                    return DataLoader(
                        self.train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        drop_last=True
                    )
        else:
            targets = extract_targets(dataset)
            if targets is not None:
                all_targets.extend(targets)

        if len(all_targets) > 0:
            # Simple binary weighting: pain (PSPI >= 1) gets 2x weight
            all_targets = np.array(all_targets, dtype=np.float32)
            weights = np.where(all_targets >= 1.0, 2.0, 1.0).astype(np.float64)

            weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
