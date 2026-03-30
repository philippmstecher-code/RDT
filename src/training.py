"""
Training loops and utilities for representational development experiments
"""
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json
import time
import random
from typing import List, Dict, Optional, Callable, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from models import create_network
from initialization import load_weights


def _clear_memory_cache():
    """Clear memory cache for any available device (CUDA or MPS)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class RemappedDataset:
    """Dataset wrapper that remaps labels to a new index range."""

    def __init__(self, subset, mapping):
        self.subset = subset
        self.mapping = mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        new_label = self.mapping[label]
        return image, new_label


class NoisyLabelDataset:
    """Dataset wrapper that applies structured label noise during training.

    noise_type:
        'within_sc'  — flip to random class in same superclass
        'between_sc' — flip to random class from different superclass
        'random'     — flip to any random class
    """

    def __init__(self, base_dataset, noise_type, noise_prob,
                 superclass_map, num_classes):
        """
        Args:
            base_dataset: underlying dataset (RemappedDataset)
            noise_type: 'within_sc' | 'between_sc' | 'random'
            noise_prob: probability of flipping each label (0.0–1.0)
            superclass_map: dict {class_idx: superclass_name}
            num_classes: total number of fine classes
        """
        self.base = base_dataset
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.num_classes = num_classes

        # Build lookup tables from superclass_map
        # sc_members[sc_name] = [class_idx, ...]
        self.sc_members = {}
        self.class_to_sc = {}
        for ci, sc_name in superclass_map.items():
            self.class_to_sc[ci] = sc_name
            self.sc_members.setdefault(sc_name, []).append(ci)

        # For between_sc: classes NOT in same superclass
        self.other_sc_classes = {}
        all_classes = set(range(num_classes))
        for ci in range(num_classes):
            sc = self.class_to_sc.get(ci)
            if sc:
                self.other_sc_classes[ci] = list(
                    all_classes - set(self.sc_members[sc])
                )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]

        if random.random() < self.noise_prob:
            if self.noise_type == 'within_sc':
                sc = self.class_to_sc.get(label)
                if sc:
                    siblings = [c for c in self.sc_members[sc] if c != label]
                    if siblings:
                        label = random.choice(siblings)

            elif self.noise_type == 'between_sc':
                others = self.other_sc_classes.get(label)
                if others:
                    label = random.choice(others)

            elif self.noise_type == 'random':
                label = random.randint(0, self.num_classes - 1)

        return image, label


def get_dataset(dataset_id: str, selected_classes: List[str], train: bool = True, experiment_id: Optional[str] = None, transform_config: Optional[Dict] = None):
    """
    Load dataset with specified classes.

    Args:
        dataset_id: Dataset identifier (e.g., 'cifar10', 'cifar100')
        selected_classes: List of class names to include
        train: Whether to load training or test split
        experiment_id: Experiment ID for locating experiment-specific dataset
        transform_config: Optional dict with keys resize, center_crop, normalize_mean, normalize_std.
                          If provided, overrides the dataset-based default transform.

    Returns:
        Dataset object and class name mapping
    """
    if experiment_id is None:
        raise ValueError("experiment_id is required for dataset loading")

    if transform_config is not None:
        # Build transform pipeline from explicit config
        steps = []

        # Resize and center crop FIRST (before augmentation) so images are the right size
        if transform_config.get('resize'):
            steps.append(transforms.Resize(transform_config['resize']))
        if transform_config.get('center_crop'):
            steps.append(transforms.CenterCrop(transform_config['center_crop']))

        # Data augmentation (training only)
        if train and transform_config.get('augmentation_enabled', True):
            # Determine image size for RandomCrop
            if transform_config.get('center_crop'):
                crop_size = transform_config['center_crop']
            elif transform_config.get('resize'):
                crop_size = transform_config['resize']
            elif dataset_id in ['cifar10', 'cifar100']:
                crop_size = 32
            else:
                crop_size = 224

            padding = transform_config.get('random_crop_padding', 4)
            if padding and padding > 0:
                steps.append(transforms.RandomCrop(crop_size, padding=padding))

            if transform_config.get('horizontal_flip', True):
                steps.append(transforms.RandomHorizontalFlip())

            brightness = transform_config.get('color_jitter_brightness')
            contrast = transform_config.get('color_jitter_contrast')
            saturation = transform_config.get('color_jitter_saturation')
            if any(v is not None and v > 0 for v in [brightness, contrast, saturation]):
                steps.append(transforms.ColorJitter(
                    brightness=brightness or 0,
                    contrast=contrast or 0,
                    saturation=saturation or 0,
                ))
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(
            mean=transform_config.get('normalize_mean', [0.5, 0.5, 0.5]),
            std=transform_config.get('normalize_std', [0.5, 0.5, 0.5])
        ))
        transform = transforms.Compose(steps)
    elif dataset_id in ['cifar10', 'cifar100']:
        if train:
            # CIFAR training: standard augmentation + normalize
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            # CIFAR validation/test: no augmentation
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    elif dataset_id == 'tiny_imagenet':
        # Tiny ImageNet: 64x64 images
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    else:
        # Default for larger datasets: resize + crop to 224, ImageNet normalization
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load dataset from experiment-specific directory
    data_root = Path(__file__).parent.parent.parent.parent / "data" / "experiments" / experiment_id / "datasets"

    if dataset_id == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_root / 'cifar10'),
            train=train,
            download=False,  # Dataset must be pre-downloaded via download manager
            transform=transform
        )
        class_names = dataset.classes
    elif dataset_id == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=str(data_root / 'cifar100'),
            train=train,
            download=False,  # Dataset must be pre-downloaded via download manager
            transform=transform
        )
        class_names = dataset.classes
    elif dataset_id == 'tiny_imagenet':
        # Tiny ImageNet: stored as ImageFolder with wnid directories
        # Restructured to: train/{wnid}/images/*.JPEG and val/{wnid}/*.JPEG
        split_name = 'train' if train else 'val'
        ti_root = data_root / 'tiny_imagenet' / split_name
        if not ti_root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet {split_name} split not found at {ti_root}. "
                f"Download via the dataset download manager."
            )
        dataset = torchvision.datasets.ImageFolder(
            root=str(ti_root),
            transform=transform
        )
        # ImageFolder uses wnid directory names as class names;
        # map them to human-readable names for the pipeline
        from tiny_imagenet_hierarchy import WNID_TO_CLASS
        class_names = [WNID_TO_CLASS.get(wnid, wnid) for wnid in dataset.classes]
        # Build a wnid->new_name mapping to update dataset internals
        wnid_to_readable = {wnid: WNID_TO_CLASS.get(wnid, wnid) for wnid in dataset.classes}
        # Update the dataset's class_to_idx to use readable names
        new_class_to_idx = {}
        for wnid, idx in dataset.class_to_idx.items():
            readable = wnid_to_readable.get(wnid, wnid)
            new_class_to_idx[readable] = idx
        dataset.class_to_idx = new_class_to_idx
        dataset.classes = class_names
    elif dataset_id == 'inat_family_genus':
        # iNaturalist family->genus: stored as ImageFolder structure
        # {family}/{genus}/img_00001.jpg
        inat_root = data_root / 'inat_family_genus'
        if not inat_root.exists():
            raise FileNotFoundError(
                f"iNaturalist dataset not found at {inat_root}. "
                f"Run download_inat.py and ensure the dataset is copied to the experiment."
            )
        dataset = torchvision.datasets.ImageFolder(
            root=str(inat_root),
            transform=transform
        )
        class_names = dataset.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")

    # Filter to selected classes
    # Handle class names with dataset prefix (e.g., "cifar100_beaver" -> "beaver")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    selected_indices = []
    for cls in selected_classes:
        # Try exact match first
        if cls in class_to_idx:
            selected_indices.append(class_to_idx[cls])
        else:
            # Try stripping dataset prefix
            stripped_cls = cls.replace(f"{dataset_id}_", "")
            if stripped_cls in class_to_idx:
                selected_indices.append(class_to_idx[stripped_cls])

    # Create mapping from dataset class index to new class index
    original_to_new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_indices)}

    # Filter dataset — use .targets attribute when available for fast filtering
    selected_set = set(selected_indices)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        indices = [i for i, label in enumerate(targets) if label in selected_set]
    else:
        indices = [i for i, (_, label) in enumerate(dataset) if label in selected_set]
    filtered_dataset = Subset(dataset, indices)

    # Wrap to remap labels using the module-level RemappedDataset class
    remapped_dataset = RemappedDataset(filtered_dataset, original_to_new)

    return remapped_dataset, selected_classes


def compute_random_baseline(dataset, num_classes: int) -> Dict[str, Any]:
    """
    Compute random baseline accuracy from actual dataset class distribution.

    Args:
        dataset: PyTorch dataset with samples and labels
        num_classes: Number of classes in the classification task

    Returns:
        Dictionary containing:
            - random_baseline: Expected accuracy of random classifier (%)
            - class_distribution: Probability of each class (sum to 1.0)
            - class_counts: Number of samples per class
            - is_balanced: Whether dataset is balanced (within 5% tolerance)
    """
    from collections import Counter

    # Count samples per class
    class_counts = Counter()

    # Handle different dataset types
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Subset of dataset with targets attribute (e.g., CIFAR, MNIST)
        for idx in dataset.indices:
            label = dataset.dataset.targets[idx]
            class_counts[label] += 1
    elif hasattr(dataset, 'targets'):
        # Direct dataset with targets attribute
        for label in dataset.targets:
            class_counts[label] += 1
    else:
        # Fallback: iterate through dataset
        for _, label in dataset:
            class_counts[label] += 1

    # Ensure all classes are represented (even if count is 0)
    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0

    # Sort by class index
    sorted_counts = [class_counts[i] for i in range(num_classes)]
    total_samples = sum(sorted_counts)

    # Compute class probabilities
    class_distribution = [count / total_samples for count in sorted_counts]

    # Compute random baseline: sum(p_i^2) for class probabilities
    # This is the expected accuracy of a classifier that outputs class i with probability p_i
    random_baseline = sum(p * p for p in class_distribution) * 100.0

    # Check if dataset is balanced (all classes within 5% of uniform distribution)
    uniform_prob = 1.0 / num_classes
    tolerance = 0.05
    is_balanced = all(abs(p - uniform_prob) <= tolerance for p in class_distribution)

    return {
        'random_baseline': random_baseline,
        'class_distribution': class_distribution,
        'class_counts': sorted_counts,
        'is_balanced': is_balanced,
    }


def create_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from configuration.

    Handles None values from Pydantic model_dump() by falling back to PyTorch defaults.
    """
    opt_type = optimizer_config['type']
    lr = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay') or 0.0

    # Helper to get config value with fallback for None (model_dump() preserves None keys)
    def _get(key, default):
        val = optimizer_config.get(key)
        return val if val is not None else default

    if opt_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=_get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=_get('nesterov', False)
        )
    elif opt_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(_get('beta1', 0.9), _get('beta2', 0.999)),
            eps=_get('epsilon', 1e-8),
            weight_decay=weight_decay
        )
    elif opt_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(_get('beta1', 0.9), _get('beta2', 0.999)),
            eps=_get('epsilon', 1e-8),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> tuple[float, float]:
    """
    Train for one epoch.

    Args:
        progress_callback: Optional callback(batch_idx, total_batches, current_loss, current_acc)
                          called periodically during training

    Returns:
        (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(train_loader)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Call progress callback on every batch for frequent updates
        if progress_callback:
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total if total > 0 else 0.0
            progress_callback(batch_idx + 1, total_batches, current_loss, current_acc)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """
    Validate model.

    Returns:
        (average_loss, accuracy)
    """
    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    # Restore original training mode
    if was_training:
        model.train()

    # Clear memory cache to prevent memory accumulation
    _clear_memory_cache()

    return avg_loss, accuracy


def _reduce_activation_dims(batch_acts: torch.Tensor, pool_method: str = "cls") -> torch.Tensor:
    """Reduce multi-dimensional activations to [batch, features].

    - 3D [batch, tokens, hidden] (Transformer):
        pool_method='cls' → extract CLS token (index 0) (ViT)
        pool_method='mean' → mean pool over tokens (CCT, no CLS token)
    - 4D [batch, channels, H, W] (CNN): global average pool over spatial dims
    - >4D: flatten
    """
    if len(batch_acts.shape) == 3:
        if pool_method == "mean":
            return batch_acts.mean(dim=1).contiguous()
        # Default: CLS token (index 0)
        # .contiguous() breaks storage sharing so torch.save doesn't serialize all tokens
        return batch_acts[:, 0, :].contiguous()
    elif len(batch_acts.shape) == 4:
        # CNN: [batch, channels, H, W] → global avg pool
        return batch_acts.mean(dim=(2, 3))
    elif len(batch_acts.shape) > 2:
        return batch_acts.reshape(batch_acts.size(0), -1)
    return batch_acts


def _get_pool_method(model: nn.Module) -> str:
    """Return the activation pooling method for this model architecture."""
    return getattr(model, '_activation_pool', 'cls')


def _is_cct(model: nn.Module) -> bool:
    """Check if the model is a CCT architecture."""
    return getattr(model, '_is_cct', False)


def extract_activations(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int
) -> Dict[str, torch.Tensor]:
    """
    Extract mean activation vectors per class from the final layer before classification.

    Returns:
        Dictionary mapping layer names to mean activation tensors per class
    """
    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    # Hook to capture activations from the layer before the final classifier
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on key layers
    # For CCT: layer norm before attention pool
    # For ResNet: layer before fc
    # For VGG: layer before classifier
    # For ViT: layer before heads
    # For EfficientNet: layer before classifier
    pool_method = _get_pool_method(model)

    if _is_cct(model):  # CCT
        hooks.append(model.norm.register_forward_hook(get_activation('features')))
    elif hasattr(model, 'fc'):  # ResNet
        hooks.append(model.avgpool.register_forward_hook(get_activation('features')))
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
        hooks.append(model.features.register_forward_hook(get_activation('features')))
    elif hasattr(model, 'heads'):  # ViT
        # Get the layer before the head
        hooks.append(model.encoder.ln.register_forward_hook(get_activation('features')))
    else:  # EfficientNet or others
        # Try to find the avgpool or adaptive pool layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(get_activation('features')))
                break

    # Incremental mean computation to avoid unbounded list accumulation
    class_mean_accumulator = {i: {'sum': None, 'count': 0} for i in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Get the captured activation
            if 'features' in activations:
                batch_activations = activations['features']

                # Reduce dimensions (CLS token for ViT, mean pool for CCT, avg pool for CNN)
                batch_activations = _reduce_activation_dims(batch_activations, pool_method)

                # Accumulate sum per class (incremental mean)
                for i, label in enumerate(labels):
                    class_idx = label.item()
                    act = batch_activations[i].cpu()

                    if class_mean_accumulator[class_idx]['sum'] is None:
                        class_mean_accumulator[class_idx]['sum'] = act.clone()
                    else:
                        class_mean_accumulator[class_idx]['sum'] += act

                    class_mean_accumulator[class_idx]['count'] += 1
                    del act

            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                _clear_memory_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute final means
    mean_activations = {}
    for class_idx in range(num_classes):
        if class_mean_accumulator[class_idx]['count'] > 0:
            mean_activations[class_idx] = (
                class_mean_accumulator[class_idx]['sum'] /
                class_mean_accumulator[class_idx]['count']
            )
        else:
            # No samples for this class - use zeros
            # Infer feature dim from other classes
            feature_dim = 1
            for other_class in range(num_classes):
                if class_mean_accumulator[other_class]['sum'] is not None:
                    feature_dim = class_mean_accumulator[other_class]['sum'].shape[0]
                    break
            mean_activations[class_idx] = torch.zeros(feature_dim)

    # Restore original training mode
    if was_training:
        model.train()

    # Clear memory cache to prevent memory accumulation
    _clear_memory_cache()

    return mean_activations


def extract_individual_activations(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    samples_per_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract individual activation vectors per sample, grouped by class.

    Returns:
        {
            'individual': {class_idx: torch.Tensor of shape [N_samples, feature_dim]},
            'mean': {class_idx: torch.Tensor},
            'std': {class_idx: torch.Tensor},
            'count': {class_idx: int}
        }
    """
    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    # Hook to capture activations from the layer before the final classifier
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on key layers (same as extract_activations)
    pool_method = _get_pool_method(model)
    if _is_cct(model):  # CCT
        hooks.append(model.norm.register_forward_hook(get_activation('features')))
    elif hasattr(model, 'fc'):  # ResNet
        hooks.append(model.avgpool.register_forward_hook(get_activation('features')))
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
        hooks.append(model.features.register_forward_hook(get_activation('features')))
    elif hasattr(model, 'heads'):  # ViT
        hooks.append(model.encoder.ln.register_forward_hook(get_activation('features')))
    else:  # EfficientNet or others
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(get_activation('features')))
                break

    # Collect activations per class
    class_activations = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Get the captured activation
            if 'features' in activations:
                batch_activations = activations['features']

                # Reduce dimensions (CLS token for ViT, mean pool for CCT, avg pool for CNN)
                batch_activations = _reduce_activation_dims(batch_activations, pool_method)

                # Group by class
                for i, label in enumerate(labels):
                    class_idx = label.item()

                    # Only collect if within samples_per_class limit
                    if samples_per_class is None or class_counts[class_idx] < samples_per_class:
                        class_activations[class_idx].append(batch_activations[i].cpu().clone())
                        class_counts[class_idx] += 1

            # Early exit when all classes reach limit
            if samples_per_class is not None:
                if all(count >= samples_per_class for count in class_counts.values()):
                    break

            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                _clear_memory_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Stack individual activations and compute statistics
    result = {
        'individual': {},
        'mean': {},
        'std': {},
        'count': {}
    }

    for class_idx in range(num_classes):
        if class_activations[class_idx]:
            # Stack to create [N_samples, feature_dim] tensor
            stacked = torch.stack(class_activations[class_idx])
            result['individual'][class_idx] = stacked
            result['mean'][class_idx] = stacked.mean(dim=0)
            result['std'][class_idx] = stacked.std(dim=0)
            result['count'][class_idx] = len(class_activations[class_idx])

            # Progressive cleanup - delete list immediately after stacking
            del class_activations[class_idx]
        else:
            # No samples for this class
            feature_dim = 1
            # Infer feature dim from other classes if possible
            for other_class in range(num_classes):
                if class_activations[other_class]:
                    feature_dim = class_activations[other_class][0].shape[0]
                    break
            result['individual'][class_idx] = torch.zeros(0, feature_dim)
            result['mean'][class_idx] = torch.zeros(feature_dim)
            result['std'][class_idx] = torch.zeros(feature_dim)
            result['count'][class_idx] = 0

    # Restore original training mode
    if was_training:
        model.train()

    # Clear memory cache to prevent memory accumulation
    _clear_memory_cache()

    return result


def _select_sae_layers(model: nn.Module, max_dim: int = 4096) -> List[str]:
    """
    Select layers suitable for SAE training based on architecture.

    Skips very early layers (relu, maxpool) and layers with flattened
    dimensions exceeding max_dim (too expensive for SAE expansion).

    Returns list of layer names to hook.
    """
    candidates = []

    if _is_cct(model):  # CCT
        num_blocks = len(model.transformer.layers)
        indices = {0, num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4, num_blocks - 1}
        for i in sorted(indices):
            candidates.append(f'transformer.layers.{i}')
        candidates.append('norm')
    elif hasattr(model, 'fc'):  # ResNet-like
        for name, module in model.named_modules():
            if 'layer' in name and name.count('.') == 0:
                candidates.append(name)
        candidates.append('avgpool')
    elif hasattr(model, 'avgpool') and not hasattr(model, 'fc'):  # EfficientNet
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                candidates.append(name)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
        for i, module in enumerate(model.features):
            if isinstance(module, nn.MaxPool2d):
                candidates.append(f'features.{i}')
    elif hasattr(model, 'heads'):  # ViT
        num_blocks = len(model.encoder.layers)
        # Select subset: first, middle, last blocks + layer norm
        indices = {0, num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4, num_blocks - 1}
        for i in sorted(indices):
            candidates.append(f'encoder.layers.{i}')
        candidates.append('encoder.ln')
    else:  # EfficientNet, etc.
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                candidates.append(name)

    return candidates


def extract_individual_multilayer_activations(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    samples_per_class: Optional[int] = None,
    selected_layers: Optional[List[str]] = None,
    max_dim: int = 4096,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract individual activation vectors per sample at multiple layers.

    Unlike extract_multilayer_activations() which returns class means,
    this returns individual samples — required for SAE training.

    Args:
        model: Model to extract from.
        data_loader: Validation data loader.
        device: Device for inference.
        num_classes: Number of classes.
        samples_per_class: Max samples per class (default: all available).
        selected_layers: Specific layers to extract. If None, auto-selected.
        max_dim: Skip layers with flattened dim > max_dim.

    Returns:
        {layer_name: {class_idx: Tensor[N_samples, feature_dim]}}
    """
    was_training = model.training
    model.eval()

    # Determine which layers to hook
    if selected_layers is None:
        layer_hooks = _select_sae_layers(model, max_dim)
    else:
        layer_hooks = selected_layers

    if not layer_hooks:
        if was_training:
            model.train()
        return {}

    # Register hooks
    activations = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach()
        return hook

    for layer_path in layer_hooks:
        parts = layer_path.split('.')
        module = model
        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            hooks.append(module.register_forward_hook(make_hook(layer_path)))
        except (AttributeError, IndexError):
            # Layer not found in this architecture, skip
            continue

    # Collect per-layer per-class individual activations
    layer_class_acts: Dict[str, Dict[int, List[torch.Tensor]]] = {
        lp: {i: [] for i in range(num_classes)} for lp in layer_hooks
    }
    class_counts = {i: 0 for i in range(num_classes)}

    # Track which layers have dimensions too large
    skipped_layers: set = set()
    pool_method = _get_pool_method(model)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model(inputs)  # triggers hooks

            for layer_name in layer_hooks:
                if layer_name in skipped_layers:
                    continue
                if layer_name not in activations:
                    continue

                batch_acts = activations[layer_name]
                # Reduce dimensions (CLS token for ViT, mean pool for CCT, avg pool for CNN)
                batch_acts = _reduce_activation_dims(batch_acts, pool_method)

                # Check dimension on first batch
                if batch_idx == 0 and batch_acts.shape[1] > max_dim:
                    skipped_layers.add(layer_name)
                    continue

                for i, label in enumerate(labels):
                    class_idx = label.item()
                    if samples_per_class is None or class_counts.get(class_idx, 0) < samples_per_class:
                        layer_class_acts[layer_name][class_idx].append(
                            batch_acts[i].cpu().clone()
                        )

            # Update class counts (use any non-skipped layer)
            for i, label in enumerate(labels):
                class_idx = label.item()
                if samples_per_class is None or class_counts[class_idx] < samples_per_class:
                    class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

            # Early exit when all classes reach limit
            if samples_per_class is not None:
                if all(count >= samples_per_class for count in class_counts.values()):
                    break

            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                _clear_memory_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Stack into tensors
    result: Dict[str, Dict[int, torch.Tensor]] = {}
    for layer_name in layer_hooks:
        if layer_name in skipped_layers:
            continue
        layer_result = {}
        has_data = False
        for class_idx in range(num_classes):
            samples = layer_class_acts[layer_name][class_idx]
            if samples:
                layer_result[class_idx] = torch.stack(samples)
                has_data = True
            else:
                # Infer feature dim from other classes
                feature_dim = 1
                for other_idx in range(num_classes):
                    other_samples = layer_class_acts[layer_name][other_idx]
                    if other_samples:
                        feature_dim = other_samples[0].shape[0]
                        break
                layer_result[class_idx] = torch.zeros(0, feature_dim)
        if has_data:
            result[layer_name] = layer_result

    # Restore training mode
    if was_training:
        model.train()

    _clear_memory_cache()
    return result


def extract_multilayer_activations(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    use_disk: bool = False,
    temp_dir: Optional[Path] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract activations from ALL layers, not just final layer.

    Args:
        model: Model to extract from
        data_loader: Data loader
        device: Device to run on
        num_classes: Number of classes
        use_disk: If True, stream to disk to reduce RAM usage (~30GB savings)
        temp_dir: Directory for temporary storage (required if use_disk=True)

    Returns:
        {
            layer_name: {class_idx: mean_activation_tensor}
        }
    """
    # Use disk-based version if requested
    if use_disk:
        if temp_dir is None:
            raise ValueError("temp_dir required when use_disk=True")
        return extract_multilayer_activations_disk(model, data_loader, device, num_classes, temp_dir)
    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    # Identify layers to hook based on architecture
    layer_hooks = []
    layer_names = []

    # Register hooks on multiple layers based on architecture
    pool_method = _get_pool_method(model)
    if _is_cct(model):  # CCT
        for i in range(len(model.transformer.layers)):
            layer_hooks.append(f'transformer.layers.{i}')
        layer_hooks.append('norm')
    elif hasattr(model, 'fc'):  # ResNet-like architectures
        # Hook into each layer group and avgpool
        for name, module in model.named_modules():
            if 'layer' in name and name.count('.') == 0:  # layer1, layer2, layer3, layer4
                layer_hooks.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                layer_hooks.append('avgpool')
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
        # Hook after each maxpool in features
        for i, module in enumerate(model.features):
            if isinstance(module, nn.MaxPool2d):
                layer_hooks.append(f'features.{i}')
    elif hasattr(model, 'heads'):  # ViT
        # Hook into each transformer block
        for i, block in enumerate(model.encoder.layers):
            layer_hooks.append(f'encoder.layers.{i}')
        layer_hooks.append('encoder.ln')
    else:  # Try to identify key layers programmatically
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                layer_hooks.append(name)

    # Storage for activations
    activations = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach()
        return hook

    # Register all hooks
    for layer_path in layer_hooks:
        parts = layer_path.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        hooks.append(module.register_forward_hook(make_hook(layer_path)))
        layer_names.append(layer_path)

    # Collect activations per layer per class
    layer_class_activations = {layer: {i: [] for i in range(num_classes)} for layer in layer_names}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass triggers all hooks
            outputs = model(inputs)

            # Store activations for each layer
            for layer_name in layer_names:
                if layer_name in activations:
                    batch_acts = activations[layer_name]

                    # Reduce dimensions (CLS token for ViT, mean pool for CCT, avg pool for CNN)
                    batch_acts = _reduce_activation_dims(batch_acts, pool_method)

                    # Group by class
                    for i, label in enumerate(labels):
                        class_idx = label.item()
                        layer_class_activations[layer_name][class_idx].append(batch_acts[i].cpu())

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute mean activations per layer per class
    result = {}
    for layer_name in layer_names:
        result[layer_name] = {}
        for class_idx in range(num_classes):
            if layer_class_activations[layer_name][class_idx]:
                result[layer_name][class_idx] = torch.stack(
                    layer_class_activations[layer_name][class_idx]
                ).mean(dim=0)
            else:
                # No samples for this class — infer dim from any populated class
                feature_dim = 1
                for other_idx in range(num_classes):
                    if layer_class_activations[layer_name][other_idx]:
                        feature_dim = layer_class_activations[layer_name][other_idx][0].shape
                        break
                result[layer_name][class_idx] = torch.zeros(feature_dim)

    # Restore original training mode
    if was_training:
        model.train()

    return result


def extract_multilayer_activations_disk(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    temp_dir: Path,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract mean activations per class from multiple layers with disk storage.

    Identical functionality to extract_multilayer_activations but streams to disk
    to avoid RAM accumulation (~30GB savings for large models).

    Args:
        model: Model to extract from
        data_loader: Data loader
        device: Device to run on
        num_classes: Number of classes
        temp_dir: Temporary directory for storing activations

    Returns:
        {
            layer_name: {class_idx: mean_activation_tensor}
        }
    """
    import shutil

    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    # Identify layers to hook (same logic as original function)
    layer_hooks = []
    layer_names = []
    pool_method = _get_pool_method(model)

    if _is_cct(model):  # CCT
        for i in range(len(model.transformer.layers)):
            layer_hooks.append(f'transformer.layers.{i}')
        layer_hooks.append('norm')
    elif hasattr(model, 'fc'):  # ResNet-like
        for name, module in model.named_modules():
            if 'layer' in name and name.count('.') == 0:
                layer_hooks.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                layer_hooks.append('avgpool')
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
        for i, module in enumerate(model.features):
            if isinstance(module, nn.MaxPool2d):
                layer_hooks.append(f'features.{i}')
    elif hasattr(model, 'heads'):  # ViT
        for i, block in enumerate(model.encoder.layers):
            layer_hooks.append(f'encoder.layers.{i}')
        layer_hooks.append('encoder.ln')
    else:
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                layer_hooks.append(name)

    # Setup hooks
    activations = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach()
        return hook

    for layer_path in layer_hooks:
        try:
            parts = layer_path.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            hooks.append(module.register_forward_hook(make_hook(layer_path)))
            layer_names.append(layer_path)
        except (AttributeError, IndexError):
            continue

    if not layer_names:
        if was_training:
            model.train()
        return {}

    # Create temp directory structure: layer_name/class_idx/batch_N.pt
    temp_dir.mkdir(parents=True, exist_ok=True)
    layer_dirs = {}
    for layer in layer_names:
        layer_dir = temp_dir / layer.replace('.', '_')
        layer_dir.mkdir(exist_ok=True)
        layer_dirs[layer] = layer_dir
        for class_idx in range(num_classes):
            (layer_dir / str(class_idx)).mkdir(exist_ok=True)

    # Stream activations to disk by class
    batch_counters = {layer: {i: 0 for i in range(num_classes)} for layer in layer_names}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # For each layer, save activations grouped by class
            for layer_name in layer_names:
                if layer_name in activations:
                    batch_acts = activations[layer_name]

                    # Reduce dimensions (CLS token for ViT, mean pool for CCT, avg pool for CNN)
                    batch_acts = _reduce_activation_dims(batch_acts, pool_method)

                    # Group by class and save each class's activations
                    for i, label in enumerate(labels):
                        class_idx = label.item()
                        class_acts = batch_acts[i:i+1].cpu().clone()
                        # Save to disk (.clone() detaches from shared storage so torch.save
                        # only serializes the actual slice, not the full batch tensor)
                        batch_num = batch_counters[layer_name][class_idx]
                        save_path = layer_dirs[layer_name] / str(class_idx) / f"batch_{batch_num}.pt"
                        torch.save(class_acts, save_path)
                        batch_counters[layer_name][class_idx] += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Now compute means from disk-stored activations
    result = {}
    for layer_name in layer_names:
        result[layer_name] = {}
        for class_idx in range(num_classes):
            class_dir = layer_dirs[layer_name] / str(class_idx)
            batch_files = sorted(class_dir.glob("batch_*.pt"))

            if batch_files:
                # Incremental mean computation to avoid loading all batches at once
                running_mean = None
                total_count = 0

                for batch_file in batch_files:
                    batch_acts = torch.load(batch_file, map_location='cpu')
                    batch_size = batch_acts.shape[0]
                    batch_mean = batch_acts.mean(dim=0)

                    if running_mean is None:
                        running_mean = batch_mean
                        total_count = batch_size
                    else:
                        # Incremental mean: new_mean = old_mean + (batch_mean - old_mean) * (batch_size / total_count)
                        total_count += batch_size
                        delta = batch_mean - running_mean
                        running_mean += delta * (batch_size / total_count)

                    del batch_acts, batch_mean

                result[layer_name][class_idx] = running_mean
            else:
                # No samples for this class — infer dim from other classes
                feature_dim = 1
                for other_idx in range(num_classes):
                    if result[layer_name].get(other_idx) is not None:
                        feature_dim = result[layer_name][other_idx].shape[0]
                        break
                result[layer_name][class_idx] = torch.zeros(feature_dim)

    # Cleanup temp files
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp files: {e}")

    if was_training:
        model.train()

    return result


def _detect_probe_layers(model: nn.Module) -> List[str]:
    """
    Intelligently detect meaningful layers to probe based on architecture.

    Strategy:
    - ResNet/Inception: Probe after each residual block
    - ViT/Transformer: Probe after each transformer block
    - VGG: Probe after each conv+pool sequence
    - EfficientNet/MobileNet: Probe after each inverted residual block
    - DenseNet: Probe after each dense block
    - ConvNeXt: Probe after each ConvNeXt block
    - Swin Transformer: Probe after each Swin block
    - MaxViT: Probe after each MaxViT block
    - UNet: Probe at encoder/decoder stages
    - Custom: Probe after significant modules (LIMITED to avoid explosion)

    Returns:
        List of layer names (module paths) to probe
    """
    probe_layers = []

    # CCT (Compact Convolutional Transformer)
    if _is_cct(model):
        print("Detected CCT architecture")
        probe_layers.append('tokenizer')
        for i in range(len(model.transformer.layers)):
            probe_layers.append(f'transformer.layers.{i}')
        probe_layers.append('norm')
        return probe_layers

    # ResNet-style (ResNet, Inception)
    if hasattr(model, 'fc') and any(hasattr(model, f'layer{i}') for i in range(1, 5)):
        print("Detected ResNet/Inception architecture")

        if hasattr(model, 'conv1'):
            probe_layers.append('conv1')
        if hasattr(model, 'maxpool') and isinstance(model.maxpool, (nn.MaxPool2d, nn.MaxPool3d)):
            probe_layers.append('maxpool')

        # Probe each residual block
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for i, block in enumerate(layer):
                    probe_layers.append(f'{layer_name}.{i}')

        if hasattr(model, 'avgpool'):
            probe_layers.append('avgpool')

    # Vision Transformer (ViT)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        print("Detected Vision Transformer architecture")

        for i, block in enumerate(model.encoder.layers):
            probe_layers.append(f'encoder.layers.{i}')

        if hasattr(model.encoder, 'ln'):
            probe_layers.append('encoder.ln')
        if hasattr(model, 'heads'):
            probe_layers.append('heads')

    # Swin Transformer
    elif hasattr(model, 'features') and any('swin' in str(type(m)).lower() for m in model.modules()):
        print("Detected Swin Transformer architecture")

        # Probe each stage in features
        if hasattr(model, 'features'):
            for i, stage in enumerate(model.features):
                # Each stage contains multiple blocks
                if hasattr(stage, '__len__'):
                    for j in range(len(stage)):
                        probe_layers.append(f'features.{i}.{j}')
                else:
                    probe_layers.append(f'features.{i}')

        if hasattr(model, 'norm'):
            probe_layers.append('norm')
        if hasattr(model, 'head'):
            probe_layers.append('head')

    # DenseNet
    elif hasattr(model, 'features') and any('denseblock' in name for name, _ in model.named_modules()):
        print("Detected DenseNet architecture")

        for name, module in model.named_modules():
            # Probe after each dense block and transition
            if 'denseblock' in name or 'transition' in name:
                if name.count('.') == 1:  # Top-level blocks only
                    probe_layers.append(name)

        if hasattr(model, 'features') and hasattr(model.features, 'norm5'):
            probe_layers.append('features.norm5')

    # EfficientNet / MobileNet
    elif hasattr(model, 'features') and any('inverted' in str(type(m)).lower() for m in model.modules()):
        print("Detected EfficientNet/MobileNet architecture")

        # Probe each inverted residual block in features
        if hasattr(model, 'features'):
            for i, block in enumerate(model.features):
                # Probe every 2nd block to avoid too many probes
                if i % 2 == 0 or i == len(model.features) - 1:
                    probe_layers.append(f'features.{i}')

        if hasattr(model, 'avgpool'):
            probe_layers.append('avgpool')

    # ConvNeXt
    elif hasattr(model, 'features') and any('convnext' in str(type(m)).lower() for m in model.modules()):
        print("Detected ConvNeXt architecture")

        # Probe each ConvNeXt block
        for name, module in model.named_modules():
            if 'features' in name and name.count('.') == 2:  # features.X.Y level
                probe_layers.append(name)

        if hasattr(model, 'avgpool'):
            probe_layers.append('avgpool')

    # MaxViT
    elif any('maxvit' in str(type(m)).lower() for m in model.modules()):
        print("Detected MaxViT architecture")

        # Probe stages and blocks
        for name, module in model.named_modules():
            if 'stem' in name and name.count('.') <= 1:
                probe_layers.append(name)
            elif 'stages' in name and name.count('.') == 2:  # stages.X.Y
                probe_layers.append(name)

    # VGG
    elif hasattr(model, 'features') and hasattr(model, 'classifier') and \
         any(isinstance(m, nn.MaxPool2d) for m in model.features):
        print("Detected VGG architecture")

        for i, module in enumerate(model.features):
            if isinstance(module, nn.MaxPool2d):
                probe_layers.append(f'features.{i}')

        if isinstance(model.classifier, nn.Sequential):
            for i in range(0, len(model.classifier), 3):
                probe_layers.append(f'classifier.{i}')

    # UNet (custom) - detect by presence of inc, down1-4, up1-4 pattern
    elif all(hasattr(model, attr) for attr in ['inc', 'down1', 'down2', 'down3', 'down4']):
        print("Detected UNet architecture")

        # Probe key encoder stages
        probe_layers.extend(['inc', 'down1', 'down2', 'down3', 'down4'])

        # Probe decoder stages if they exist
        if hasattr(model, 'up1'):
            probe_layers.extend(['up1', 'up2', 'up3', 'up4'])

        # Probe the bottleneck pooling and classifier
        if hasattr(model, 'global_pool'):
            probe_layers.append('global_pool')

    # Fallback: probe ONLY pooling and top-level blocks to avoid explosion
    else:
        print("Using fallback layer detection for custom architecture")
        print("  WARNING: Limiting to depth=1 and pooling layers only to avoid excessive probing")

        for name, module in model.named_modules():
            if name == '':
                continue

            # ONLY probe pooling layers and top-level (depth=1) significant modules
            # This prevents explosion from deeply nested architectures
            depth = name.count('.')

            # Always include pooling layers at any depth (these are key architectural points)
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                if depth <= 2:  # Limit even pooling layers to reasonable depth
                    probe_layers.append(name)

            # For other layers, only include if at depth 1 (top-level modules)
            elif depth == 1 and isinstance(module, (
                nn.Sequential,
                nn.ModuleList
            )):
                probe_layers.append(name)

    # Deduplicate while preserving order
    seen = set()
    probe_layers = [x for x in probe_layers if not (x in seen or seen.add(x))]

    # SAFETY: If we detected more than 50 layers, subsample to avoid excessive computation
    if len(probe_layers) > 50:
        print(f"  WARNING: {len(probe_layers)} layers detected, subsampling to 50 evenly spaced layers")
        # Keep first and last, sample evenly from the middle
        step = len(probe_layers) // 48
        sampled = [probe_layers[0]] + probe_layers[1:-1:step][:48] + [probe_layers[-1]]
        probe_layers = sampled

    print(f"Selected {len(probe_layers)} layers to probe: {probe_layers[:10]}{'...' if len(probe_layers) > 10 else ''}")
    return probe_layers


def train_linear_probes(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_classes: int,
    use_disk: bool = False,
    temp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train linear classifiers on frozen representations from each layer.

    Optimized to extract activations from all layers in a single pass through the data,
    rather than running the model separately for each layer.

    Process:
    1. Freeze all model weights
    2. Automatically detect all meaningful layers in the architecture
    3. Register hooks for ALL layers simultaneously
    4. Extract activations for train/val sets in ONE pass per dataset
    5. For each layer, train linear classifier and evaluate

    Args:
        model: Model to probe
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        num_classes: Number of classes
        use_disk: If True, use disk-based storage to reduce RAM usage (~30GB savings)
        temp_dir: Directory for temporary activation storage (required if use_disk=True)

    Returns:
        {
            'layer_accuracies': {layer_name: {'train': float, 'val': float}},
            'layer_dimensions': {layer_name: int},
            'best_layer': str,
            'best_val_accuracy': float
        }
    """
    # Use disk-based version if requested
    if use_disk:
        if temp_dir is None:
            raise ValueError("temp_dir required when use_disk=True")
        return train_linear_probes_disk(model, train_loader, val_loader, device, num_classes, temp_dir)
    # Save original training mode and set to eval (freeze model)
    was_training = model.training
    model.eval()

    # Intelligently detect layers to probe based on architecture
    layer_paths = _detect_probe_layers(model)

    if not layer_paths:
        print("No layers detected for probing")
        if was_training:
            model.train()
        return {
            'layer_accuracies': {},
            'layer_dimensions': {},
            'best_layer': None,
            'best_val_accuracy': 0.0
        }

    print(f"Extracting activations from {len(layer_paths)} layers in parallel...")

    # Register hooks for ALL layers at once
    activations = {}
    hooks = []
    valid_layers = []
    pool_method = _get_pool_method(model)

    def make_hook(name):
        def hook(module, input, output):
            act = output.detach()
            act = _reduce_activation_dims(act, pool_method)
            activations[name] = act
        return hook

    # Register all hooks
    for layer_path in layer_paths:
        try:
            parts = layer_path.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)

            hook = module.register_forward_hook(make_hook(layer_path))
            hooks.append(hook)
            valid_layers.append(layer_path)
        except (AttributeError, IndexError) as e:
            print(f"  Skipping {layer_path}: module not found ({e})")

    if not valid_layers:
        print("No valid layers found for probing")
        if was_training:
            model.train()
        return {
            'layer_accuracies': {},
            'layer_dimensions': {},
            'best_layer': None,
            'best_val_accuracy': 0.0
        }

    # Collect activations for ALL layers in ONE pass through training data
    print("Collecting train activations...")
    train_data = {layer: [] for layer in valid_layers}
    train_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            model(inputs)

            # Store activations from all layers for this batch
            for layer_path in valid_layers:
                if layer_path in activations:
                    train_data[layer_path].append(activations[layer_path].cpu().numpy())

            train_labels.append(labels.numpy())

    train_labels = np.concatenate(train_labels, axis=0)

    # Collect activations for ALL layers in ONE pass through validation data
    print("Collecting validation activations...")
    val_data = {layer: [] for layer in valid_layers}
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            model(inputs)

            # Store activations from all layers for this batch
            for layer_path in valid_layers:
                if layer_path in activations:
                    val_data[layer_path].append(activations[layer_path].cpu().numpy())

            val_labels.append(labels.numpy())

    val_labels = np.concatenate(val_labels, axis=0)

    # Remove all hooks now that we're done collecting
    for hook in hooks:
        hook.remove()

    # Now train probes for each layer using the collected activations
    print("Training linear probes...")
    result = {
        'layer_accuracies': {},
        'layer_dimensions': {},
        'best_layer': None,
        'best_val_accuracy': 0.0
    }

    for idx, layer_path in enumerate(valid_layers):
        print(f"  [{idx+1}/{len(valid_layers)}] Training probe for layer: {layer_path}...", end='', flush=True)

        # Concatenate batches for this layer and immediately free the per-layer list
        if not train_data[layer_path] or not val_data[layer_path]:
            print(f" SKIPPED (no activations)")
            train_data.pop(layer_path, None)
            val_data.pop(layer_path, None)
            continue

        X_train = np.concatenate(train_data[layer_path], axis=0)
        del train_data[layer_path]  # Free this layer's batches immediately

        X_val = np.concatenate(val_data[layer_path], axis=0)
        del val_data[layer_path]  # Free this layer's batches immediately

        # Sanitize activations: replace NaN/Inf with 0 (can occur in early-training snapshots
        # with unstable random initializations, causing sklearn matmul warnings)
        if not np.isfinite(X_train).all():
            nan_count = (~np.isfinite(X_train)).sum()
            print(f" [warning: {nan_count} non-finite values in train activations, replacing with 0]", end='', flush=True)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(X_val).all():
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features to prevent numerical overflow in lbfgs solver
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        # Two problems after StandardScaler:
        # 1. std=0 (dead neurons) → NaN; replace with 0
        # 2. std≈0 (near-dead neurons) → huge finite values (e.g. 1e15); clip to [-10, 10]
        X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

        # Train linear probe with optimized settings
        probe = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            C=1.0,
            tol=1e-4,
            random_state=42,
            n_jobs=-1  # Use all CPU cores for faster training
        )

        try:
            probe.fit(X_train, train_labels)

            # Evaluate
            train_pred = probe.predict(X_train)
            val_pred = probe.predict(X_val)

            train_acc = accuracy_score(train_labels, train_pred) * 100
            val_acc = accuracy_score(val_labels, val_pred) * 100

            result['layer_accuracies'][layer_path] = {
                'train': train_acc,
                'val': val_acc
            }
            result['layer_dimensions'][layer_path] = X_train.shape[1]

            # Track best layer
            if val_acc > result['best_val_accuracy']:
                result['best_val_accuracy'] = val_acc
                result['best_layer'] = layer_path

            print(f" Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        except Exception as e:
            print(f" FAILED: {e}")

        # Free this layer's activations before moving to the next
        del X_train, X_val
        _clear_memory_cache()

    # Restore original training mode
    if was_training:
        model.train()

    return result


def train_linear_probes_disk(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_classes: int,
    temp_dir: Path,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Train linear classifiers on frozen representations with disk-based storage.

    This version stores activations to disk to avoid RAM accumulation (~30GB savings).
    Functionality is IDENTICAL to train_linear_probes but uses disk as intermediate storage.

    Process:
    1. Stream train activations to disk files (one file per layer)
    2. Stream val activations to disk files
    3. For each layer, load activations from disk and train probe
    4. Clean up temporary files

    Args:
        model: Trained model to probe
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        num_classes: Number of classes
        temp_dir: Temporary directory for storing activation files

    Returns:
        {
            'layer_accuracies': {layer_name: {'train': float, 'val': float}},
            'layer_dimensions': {layer_name: int},
            'best_layer': str,
            'best_val_accuracy': float
        }
    """
    import tempfile
    import shutil

    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    # Detect layers to probe
    layer_paths = _detect_probe_layers(model)
    if not layer_paths:
        print("No layers detected for probing")
        if was_training:
            model.train()
        return {
            'layer_accuracies': {},
            'layer_dimensions': {},
            'best_layer': None,
            'best_val_accuracy': 0.0
        }

    print(f"Extracting activations from {len(layer_paths)} layers with disk storage...")

    # Create temporary directory structure
    temp_dir.mkdir(parents=True, exist_ok=True)
    train_act_dir = temp_dir / "train_activations"
    val_act_dir = temp_dir / "val_activations"
    train_act_dir.mkdir(exist_ok=True)
    val_act_dir.mkdir(exist_ok=True)

    # Register hooks for all layers
    activations = {}
    hooks = []
    valid_layers = []
    pool_method = _get_pool_method(model)

    def make_hook(name):
        def hook(module, input, output):
            act = output.detach()
            act = _reduce_activation_dims(act, pool_method)
            activations[name] = act
        return hook

    # Register all hooks
    for layer_path in layer_paths:
        try:
            parts = layer_path.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            hook = module.register_forward_hook(make_hook(layer_path))
            hooks.append(hook)
            valid_layers.append(layer_path)
        except (AttributeError, IndexError):
            continue

    if not valid_layers:
        print("No valid layers found for probing")
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()
        return {
            'layer_accuracies': {},
            'layer_dimensions': {},
            'best_layer': None,
            'best_val_accuracy': 0.0
        }

    # Stream train activations to disk (append batches to .npy files)
    print("Collecting train activations (streaming to disk)...")
    train_batch_counts = {layer: 0 for layer in valid_layers}

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            model(inputs)

            # Write each layer's activations to disk immediately
            for layer_path in valid_layers:
                if layer_path in activations:
                    act_np = activations[layer_path].cpu().numpy()
                    layer_filename = layer_path.replace('.', '_')
                    # Save activations
                    batch_file = train_act_dir / f"{layer_filename}_batch_{train_batch_counts[layer_path]}.npy"
                    np.save(batch_file, act_np, allow_pickle=False)
                    # Save labels alongside activations
                    label_file = train_act_dir / f"{layer_filename}_labels_batch_{train_batch_counts[layer_path]}.npy"
                    np.save(label_file, labels.numpy(), allow_pickle=False)
                    train_batch_counts[layer_path] += 1

    # Stream validation activations to disk
    print("Collecting validation activations (streaming to disk)...")
    val_batch_counts = {layer: 0 for layer in valid_layers}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            model(inputs)

            for layer_path in valid_layers:
                if layer_path in activations:
                    act_np = activations[layer_path].cpu().numpy()
                    layer_filename = layer_path.replace('.', '_')
                    # Save activations
                    batch_file = val_act_dir / f"{layer_filename}_batch_{val_batch_counts[layer_path]}.npy"
                    np.save(batch_file, act_np, allow_pickle=False)
                    # Save labels alongside activations
                    label_file = val_act_dir / f"{layer_filename}_labels_batch_{val_batch_counts[layer_path]}.npy"
                    np.save(label_file, labels.numpy(), allow_pickle=False)
                    val_batch_counts[layer_path] += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Now train probes layer by layer, loading from disk
    print("Training linear probes from disk-stored activations...")
    result = {
        'layer_accuracies': {},
        'layer_dimensions': {},
        'best_layer': None,
        'best_val_accuracy': 0.0
    }

    for idx, layer_path in enumerate(valid_layers):
        print(f"  [{idx+1}/{len(valid_layers)}] Training probe for {layer_path}...", end='', flush=True)

        # Load activations for this layer from disk
        layer_filename = layer_path.replace('.', '_')

        try:
            # Check if files exist
            if train_batch_counts[layer_path] == 0 or val_batch_counts[layer_path] == 0:
                print(" SKIPPED (no activations)")
                continue

            # Load ALL training activations into memory (once per layer)
            print(f" [Loading {train_batch_counts[layer_path]} train batches into RAM]", end='', flush=True)
            train_batches = []
            train_label_batches = []
            for i in range(train_batch_counts[layer_path]):
                batch_file = train_act_dir / f"{layer_filename}_batch_{i}.npy"
                label_file = train_act_dir / f"{layer_filename}_labels_batch_{i}.npy"
                train_batches.append(np.load(batch_file, allow_pickle=False))
                train_label_batches.append(np.load(label_file, allow_pickle=False))

            X_train = np.concatenate(train_batches, axis=0)
            y_train = np.concatenate(train_label_batches, axis=0)
            del train_batches, train_label_batches  # Free memory

            # Load ALL validation activations into memory (once per layer)
            print(f" [Loading {val_batch_counts[layer_path]} val batches]", end='', flush=True)
            val_batches = []
            val_label_batches = []
            for i in range(val_batch_counts[layer_path]):
                batch_file = val_act_dir / f"{layer_filename}_batch_{i}.npy"
                label_file = val_act_dir / f"{layer_filename}_labels_batch_{i}.npy"
                val_batches.append(np.load(batch_file, allow_pickle=False))
                val_label_batches.append(np.load(label_file, allow_pickle=False))

            X_val = np.concatenate(val_batches, axis=0)
            y_val = np.concatenate(val_label_batches, axis=0)
            del val_batches, val_label_batches  # Free memory

            # Sanitize activations: replace NaN/Inf with 0 (can occur in early-training snapshots
            # with unstable random initializations, causing sklearn matmul warnings)
            if not np.isfinite(X_train).all():
                nan_count = (~np.isfinite(X_train)).sum()
                print(f" [warning: {nan_count} non-finite values in train activations, replacing with 0]", end='', flush=True)
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(X_val).all():
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features to prevent numerical overflow in lbfgs solver
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            # Two problems after StandardScaler:
            # 1. std=0 (dead neurons) → NaN; replace with 0
            # 2. std≈0 (near-dead neurons) → huge finite values (e.g. 1e15); clip to [-10, 10]
            X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
            X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

            probe = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                C=1.0,
                tol=1e-4,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )

            probe.fit(X_train, y_train)

            # Evaluate
            train_pred = probe.predict(X_train)
            val_pred = probe.predict(X_val)

            train_acc = accuracy_score(y_train, train_pred) * 100
            val_acc = accuracy_score(y_val, val_pred) * 100

            feature_dim = X_train.shape[1]

            # Free memory after training this layer
            del X_train, y_train, X_val, y_val, train_pred, val_pred

            result['layer_accuracies'][layer_path] = {
                'train': train_acc,
                'val': val_acc
            }
            result['layer_dimensions'][layer_path] = feature_dim

            if val_acc > result['best_val_accuracy']:
                result['best_val_accuracy'] = val_acc
                result['best_layer'] = layer_path

            print(f" Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        except Exception as e:
            print(f" ERROR: {e}")

    # Cleanup: delete temporary files
    try:
        shutil.rmtree(train_act_dir, ignore_errors=True)
        shutil.rmtree(val_act_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp files: {e}")

    # Restore original training mode
    if was_training:
        model.train()

    return result


def compute_class_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    class_names: List[str],
    num_classes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Compute per-class performance metrics.

    Args:
        model: Neural network model
        data_loader: DataLoader for evaluation
        device: Device to run on ('cuda', 'cpu', 'mps')
        class_names: List of class names
        num_classes: Number of classes (optional, derived from class_names if not provided)

    Returns:
        List of dictionaries with class metrics
    """
    # Save original training mode and set to eval
    was_training = model.training
    model.eval()

    if num_classes is None:
        num_classes = len(class_names)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_confidence = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            for i, label in enumerate(labels):
                class_idx = label.item()
                class_total[class_idx] += 1
                class_correct[class_idx] += (predicted[i] == label).item()
                class_confidence[class_idx].append(probs[i][class_idx].item())

    metrics = []
    for i in range(num_classes):
        accuracy = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        avg_conf = np.mean(class_confidence[i]) if class_confidence[i] else 0.0

        metrics.append({
            'class_name': class_names[i],
            'class_index': i,
            'accuracy': accuracy,
            'sample_count': class_total[i],
            'avg_confidence': avg_conf,
        })

    # Restore original training mode
    if was_training:
        model.train()

    return metrics
