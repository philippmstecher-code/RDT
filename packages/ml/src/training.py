"""Datasets, training loops and activation extraction."""
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
        'within_sc': flip to a random class in the same superclass
        'between_sc': flip to a random class from a different superclass
        'between_sc_matched': flip to a per-class list of out-of-superclass targets
        'random': flip to any class

    fixed:
        False (default): noise is redrawn on every __getitem__, so a sample may flip
            differently each epoch. The true label remains the modal signal across
            epochs and the noise acts as a regulariser.
        True: the corruption is drawn once per sample index (deterministically from
            `seed`) and is identical every epoch, giving static, memorisable label
            noise.
    """

    def __init__(self, base_dataset, noise_type, noise_prob,
                 superclass_map, num_classes, fixed=False, seed=0, matched_targets=None):
        """
        Args:
            base_dataset: underlying dataset (RemappedDataset)
            noise_type: 'within_sc' | 'between_sc' | 'random'
            noise_prob: probability of flipping each label (0 to 1)
            superclass_map: dict {class_idx: superclass_name}
            num_classes: total number of fine classes
            fixed: if True, corruption is static (drawn once per sample index)
                instead of re-randomised each epoch
            seed: RNG seed for the deterministic per-sample draw when fixed=True
        """
        self.base = base_dataset
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.num_classes = num_classes
        self.fixed = fixed
        self.seed = seed
        # 'between_sc_matched': per class, flip targets outside its superclass, matched in similarity
        # to its siblings ({class_idx: [targets]}).
        self.matched_targets = ({int(k): [int(x) for x in v] for k, v in matched_targets.items()}
                                if matched_targets else None)
        if noise_type == 'between_sc_matched' and not self.matched_targets:
            raise ValueError("between_sc_matched requires matched_targets")

        # sc_members[sc_name] = [class_idx, ...]
        self.sc_members = {}
        self.class_to_sc = {}
        for ci, sc_name in superclass_map.items():
            self.class_to_sc[ci] = sc_name
            self.sc_members.setdefault(sc_name, []).append(ci)

        # For between_sc: classes outside the sample's superclass
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

        # fixed=True: deterministic per-sample RNG -> same draw every epoch (static
        # corruption). fixed=False: global RNG -> re-randomised each epoch.
        rng = random.Random(self.seed * 1_000_003 + idx) if self.fixed else random

        if rng.random() < self.noise_prob:
            if self.noise_type == 'within_sc':
                sc = self.class_to_sc.get(label)
                if sc:
                    siblings = [c for c in self.sc_members[sc] if c != label]
                    if siblings:
                        label = rng.choice(siblings)

            elif self.noise_type == 'between_sc':
                others = self.other_sc_classes.get(label)
                if others:
                    label = rng.choice(others)

            elif self.noise_type == 'between_sc_matched':
                label = rng.choice(self.matched_targets[label])

            elif self.noise_type == 'random':
                label = rng.randint(0, self.num_classes - 1)

        return image, label


class SuperclassSelectiveBlurDataset:
    """Dataset wrapper that applies an input deficit (downsample or Gaussian blur) to
    a subset of superclasses (set A) during a training-epoch window.

    Half of the superclasses (set A) are blurred during the window while the other
    half (set B) stay sharp within the same network, which serves as the
    within-network control.

    The wrapper holds two views of the same selected classes, a sharp and a blurred
    dataset (both from get_dataset with identical selected_classes, so sample index i
    and its label agree across the two). __getitem__ returns the blurred view if
    blur_active is set and the label is in set A, and the sharp view otherwise. Set A
    is given as remapped fine-class indices, the label space used by the model and
    the SAE analysis.

    The mutable `blur_active` flag is switched by the training loop at the onset and
    offset epochs, so any window [onset, offset) is supported.

    The two views draw their RandomCrop/flip independently, but one view is returned
    per call, so each sample is a single augmented image. Validation uses the sharp
    pipeline.
    """

    def __init__(self, sharp_dataset, blurred_dataset, set_a_class_indices):
        """
        Args:
            sharp_dataset: get_dataset(...) result without input deficit.
            blurred_dataset: get_dataset(...) result with the same selected_classes
                and the input deficit (downsample or gaussian) in its transform.
            set_a_class_indices: remapped fine-class indices whose superclass is in
                set A (the blurred half).
        """
        if len(sharp_dataset) != len(blurred_dataset):
            raise ValueError(
                f"sharp/blurred dataset length mismatch: "
                f"{len(sharp_dataset)} vs {len(blurred_dataset)}")
        self.sharp = sharp_dataset
        self.blurred = blurred_dataset
        self.set_a = set(int(c) for c in set_a_class_indices)
        # Switched by the training loop to open/close the deficit window. Default on
        # so that, if the loop never touches it (e.g. window starts at epoch 0), the
        # deficit is applied from the first epoch.
        self.blur_active = True

    def __len__(self):
        return len(self.sharp)

    def __getitem__(self, idx):
        # Peek the label cheaply from the sharp view (same label in both views).
        image, label = self.sharp[idx]
        if self.blur_active and label in self.set_a:
            # Return the blurred view of this sample (its own augmentation draw).
            b_image, _ = self.blurred[idx]
            return b_image, label
        return image, label


def get_dataset(dataset_id: str, selected_classes: List[str], train: bool = True, experiment_id: Optional[str] = None, transform_config: Optional[Dict] = None):
    """Load a dataset restricted to the selected classes.

    Args:
        dataset_id: Dataset identifier (e.g., 'cifar10', 'cifar100')
        selected_classes: List of class names to include
        train: Whether to load training or test split
        experiment_id: Experiment ID for locating experiment-specific dataset
        transform_config: Optional dict with keys resize, center_crop, normalize_mean,
            normalize_std (and optional blur settings); overrides the default transform.

    Returns:
        Dataset object and class name mapping
    """
    if experiment_id is None:
        raise ValueError("experiment_id is required for dataset loading")

    if transform_config is not None:
        steps = []

        # Resize and center-crop before augmentation
        if transform_config.get('resize'):
            steps.append(transforms.Resize(transform_config['resize']))
        if transform_config.get('center_crop'):
            steps.append(transforms.CenterCrop(transform_config['center_crop']))

        if train and transform_config.get('augmentation_enabled', True):
            if transform_config.get('center_crop'):
                crop_size = transform_config['center_crop']
            elif transform_config.get('resize'):
                crop_size = transform_config['resize']
            elif dataset_id in ['cifar10', 'cifar100', 'tiny_imagenet']:
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
        # Gaussian blur (input-deficit curriculum): low-pass the inputs to deprive
        # high-frequency feature formation. Applied last (on the augmented image) so
        # only the spatial-frequency content is changed, not the crop/flip geometry.
        gb = transform_config.get('gaussian_blur')
        if gb:
            steps.append(transforms.GaussianBlur(
                kernel_size=gb['kernel_size'], sigma=gb['sigma']))
        # Downsample blur (Achille et al. 2019 cataract-like deficit): bilinearly
        # downscale to a low resolution then upscale back, destroying small-scale
        # detail; a stronger deficit than Gaussian low-pass.
        db = transform_config.get('downsample_blur')
        if db:
            BILINEAR = transforms.InterpolationMode.BILINEAR
            steps.append(transforms.Resize(db['size'], interpolation=BILINEAR))
            steps.append(transforms.Resize(db.get('out', 32), interpolation=BILINEAR))
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(
            mean=transform_config.get('normalize_mean', [0.5, 0.5, 0.5]),
            std=transform_config.get('normalize_std', [0.5, 0.5, 0.5])
        ))
        transform = transforms.Compose(steps)
    elif dataset_id in ['cifar10', 'cifar100']:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    elif dataset_id == 'tiny_imagenet':
        # Tiny ImageNet is resized from 64x64 to 32x32 so the CIFAR architecture
        # configs (resnet18, vit_tiny with image_size=32, cct_7) train unchanged;
        # the cross-dataset test varies the class hierarchy, not the input
        # resolution.
        if train:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    else:
        # Other datasets: resize and crop to 224, ImageNet normalisation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_root = Path(__file__).parent.parent.parent.parent / "data" / "experiments" / experiment_id / "datasets"

    if dataset_id == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_root / 'cifar10'),
            train=train,
            download=False,  # dataset must be downloaded beforehand
            transform=transform
        )
        class_names = dataset.classes
    elif dataset_id == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=str(data_root / 'cifar100'),
            train=train,
            download=False,  # dataset must be downloaded beforehand
            transform=transform
        )
        class_names = dataset.classes
    elif dataset_id == 'tiny_imagenet':
        # Tiny ImageNet: stored as ImageFolder with wnid directories
        # Layout: train/{wnid}/images/*.JPEG and val/{wnid}/*.JPEG
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
        # map them to readable names
        from tiny_imagenet_hierarchy import WNID_TO_CLASS
        class_names = [WNID_TO_CLASS.get(wnid, wnid) for wnid in dataset.classes]
        wnid_to_readable = {wnid: WNID_TO_CLASS.get(wnid, wnid) for wnid in dataset.classes}
        new_class_to_idx = {}
        for wnid, idx in dataset.class_to_idx.items():
            readable = wnid_to_readable.get(wnid, wnid)
            new_class_to_idx[readable] = idx
        dataset.class_to_idx = new_class_to_idx
        dataset.classes = class_names
    elif dataset_id == 'inat_family_genus':
        # iNaturalist family/genus, stored as an ImageFolder:
        # {family}/{genus}/img_00001.jpg
        inat_root = data_root / 'inat_family_genus'
        if not inat_root.exists():
            raise FileNotFoundError(
                f"iNaturalist dataset not found at {inat_root}. "
                f"Download the iNaturalist family/genus dataset and copy it to the experiment."
            )
        dataset = torchvision.datasets.ImageFolder(
            root=str(inat_root),
            transform=transform
        )
        class_names = dataset.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")

    # Class names may carry a dataset prefix (e.g. "cifar100_beaver" -> "beaver")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    selected_indices = []
    for cls in selected_classes:
        if cls in class_to_idx:
            selected_indices.append(class_to_idx[cls])
        else:
            stripped_cls = cls.replace(f"{dataset_id}_", "")
            if stripped_cls in class_to_idx:
                selected_indices.append(class_to_idx[stripped_cls])

    original_to_new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_indices)}

    # Use .targets when available for fast filtering
    selected_set = set(selected_indices)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        indices = [i for i, label in enumerate(targets) if label in selected_set]
    else:
        indices = [i for i, (_, label) in enumerate(dataset) if label in selected_set]
    filtered_dataset = Subset(dataset, indices)

    remapped_dataset = RemappedDataset(filtered_dataset, original_to_new)

    return remapped_dataset, selected_classes


def compute_random_baseline(dataset, num_classes: int) -> Dict[str, Any]:
    """Compute the chance-level accuracy from the dataset's class distribution.

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

    class_counts = Counter()

    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Subset of a dataset with .targets (e.g. CIFAR)
        for idx in dataset.indices:
            label = dataset.dataset.targets[idx]
            class_counts[label] += 1
    elif hasattr(dataset, 'targets'):
        # Dataset with .targets
        for label in dataset.targets:
            class_counts[label] += 1
    else:
        # Otherwise iterate through the dataset
        for _, label in dataset:
            class_counts[label] += 1

    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0

    sorted_counts = [class_counts[i] for i in range(num_classes)]
    total_samples = sum(sorted_counts)

    class_distribution = [count / total_samples for count in sorted_counts]

    # Chance accuracy sum(p_i^2): expected accuracy of a classifier that
    # outputs class i with probability p_i
    random_baseline = sum(p * p for p in class_distribution) * 100.0

    # Balanced: all class frequencies within 5% of uniform
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

    # model_dump() keeps None-valued keys, so fall back to defaults explicitly
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
    """Train for one epoch.

    Args:
        progress_callback: Optional callback(batch_idx, total_batches, current_loss,
            current_acc), called after every batch

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
    """Evaluate the model on a data loader.

    Returns:
        (average_loss, accuracy)
    """
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

    print(f"  [VALIDATE] correct={correct}, total={total}, accuracy={accuracy:.10f}%")

    if was_training:
        model.train()

    _clear_memory_cache()

    return avg_loss, accuracy


def _reduce_activation_dims(batch_acts: torch.Tensor, pool_method: str = "cls") -> torch.Tensor:
    """Reduce multi-dimensional activations to [batch, features].

    - 3D [batch, tokens, hidden] (Transformer):
        pool_method='cls': CLS token (index 0) (ViT)
        pool_method='mean': mean over tokens (CCT, which has no CLS token)
    - 4D [batch, channels, H, W] (CNN): global average pool over spatial dims
    - >4D: flatten
    """
    if len(batch_acts.shape) == 3:
        if pool_method == "mean":
            return batch_acts.mean(dim=1).contiguous()
        # .contiguous() breaks storage sharing so torch.save does not serialise all tokens
        return batch_acts[:, 0, :].contiguous()
    elif len(batch_acts.shape) == 4:
        # CNN: [batch, channels, H, W] -> global average pool
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
    """Mean activation vector per class at the layer before the classifier.

    Returns:
        Dictionary mapping layer names to mean activation tensors per class
    """
    was_training = model.training
    model.eval()

    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Hook the layer before the classifier:
    # CCT: layer norm before attention pooling; ResNet: layer before fc;
    # VGG: layer before classifier; ViT: layer before heads;
    # EfficientNet: layer before classifier
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
        # Use the avgpool or adaptive pooling layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(get_activation('features')))
                break

    # Running per-class sums instead of storing all activations
    class_mean_accumulator = {i: {'sum': None, 'count': 0} for i in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if 'features' in activations:
                batch_activations = activations['features']

                batch_activations = _reduce_activation_dims(batch_activations, pool_method)

                for i, label in enumerate(labels):
                    class_idx = label.item()
                    act = batch_activations[i].cpu()

                    if class_mean_accumulator[class_idx]['sum'] is None:
                        class_mean_accumulator[class_idx]['sum'] = act.clone()
                    else:
                        class_mean_accumulator[class_idx]['sum'] += act

                    class_mean_accumulator[class_idx]['count'] += 1
                    del act

            if batch_idx % 10 == 0:
                _clear_memory_cache()

    for hook in hooks:
        hook.remove()

    mean_activations = {}
    for class_idx in range(num_classes):
        if class_mean_accumulator[class_idx]['count'] > 0:
            mean_activations[class_idx] = (
                class_mean_accumulator[class_idx]['sum'] /
                class_mean_accumulator[class_idx]['count']
            )
        else:
            # No samples for this class: zeros of the feature dimension of the other classes
            feature_dim = 1
            for other_class in range(num_classes):
                if class_mean_accumulator[other_class]['sum'] is not None:
                    feature_dim = class_mean_accumulator[other_class]['sum'].shape[0]
                    break
            mean_activations[class_idx] = torch.zeros(feature_dim)

    if was_training:
        model.train()

    _clear_memory_cache()

    return mean_activations


def extract_individual_activations(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_classes: int,
    samples_per_class: Optional[int] = None
) -> Dict[str, Any]:
    """Per-sample activation vectors at the layer before the classifier, grouped by class.

    Returns:
        {
            'individual': {class_idx: torch.Tensor of shape [N_samples, feature_dim]},
            'mean': {class_idx: torch.Tensor},
            'std': {class_idx: torch.Tensor},
            'count': {class_idx: int}
        }
    """
    was_training = model.training
    model.eval()

    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Same hook layer as extract_activations
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

    class_activations = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if 'features' in activations:
                batch_activations = activations['features']

                batch_activations = _reduce_activation_dims(batch_activations, pool_method)

                for i, label in enumerate(labels):
                    class_idx = label.item()

                    if samples_per_class is None or class_counts[class_idx] < samples_per_class:
                        class_activations[class_idx].append(batch_activations[i].cpu().clone())
                        class_counts[class_idx] += 1

            # Stop once every class has samples_per_class samples
            if samples_per_class is not None:
                if all(count >= samples_per_class for count in class_counts.values()):
                    break

            if batch_idx % 10 == 0:
                _clear_memory_cache()

    for hook in hooks:
        hook.remove()

    result = {
        'individual': {},
        'mean': {},
        'std': {},
        'count': {}
    }

    for class_idx in range(num_classes):
        if class_activations[class_idx]:
            stacked = torch.stack(class_activations[class_idx])
            result['individual'][class_idx] = stacked
            result['mean'][class_idx] = stacked.mean(dim=0)
            result['std'][class_idx] = stacked.std(dim=0)
            result['count'][class_idx] = len(class_activations[class_idx])

            del class_activations[class_idx]
        else:
            feature_dim = 1
            # No samples for this class: infer the feature dimension from the other classes
            for other_class in range(num_classes):
                if class_activations[other_class]:
                    feature_dim = class_activations[other_class][0].shape[0]
                    break
            result['individual'][class_idx] = torch.zeros(0, feature_dim)
            result['mean'][class_idx] = torch.zeros(feature_dim)
            result['std'][class_idx] = torch.zeros(feature_dim)
            result['count'][class_idx] = 0

    if was_training:
        model.train()

    _clear_memory_cache()

    return result


def _select_sae_layers(model: nn.Module, max_dim: int = 4096) -> List[str]:
    """Select the layers used for SAE training for the given architecture.

    Skips very early layers (relu, maxpool) and layers whose flattened dimension
    exceeds max_dim.

    Returns a list of layer names to hook.
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
        # First, middle and last blocks plus the final layer norm
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
    """Per-sample activation vectors at multiple layers (input to SAE training).

    extract_multilayer_activations() returns class means instead.

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

    if selected_layers is None:
        layer_hooks = _select_sae_layers(model, max_dim)
    else:
        layer_hooks = selected_layers

    if not layer_hooks:
        if was_training:
            model.train()
        return {}

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
            continue

    layer_class_acts: Dict[str, Dict[int, List[torch.Tensor]]] = {
        lp: {i: [] for i in range(num_classes)} for lp in layer_hooks
    }
    class_counts = {i: 0 for i in range(num_classes)}

    # Layers skipped because their dimension exceeds max_dim
    skipped_layers: set = set()
    pool_method = _get_pool_method(model)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model(inputs)

            for layer_name in layer_hooks:
                if layer_name in skipped_layers:
                    continue
                if layer_name not in activations:
                    continue

                batch_acts = activations[layer_name]
                batch_acts = _reduce_activation_dims(batch_acts, pool_method)

                if batch_idx == 0 and batch_acts.shape[1] > max_dim:
                    skipped_layers.add(layer_name)
                    continue

                for i, label in enumerate(labels):
                    class_idx = label.item()
                    if samples_per_class is None or class_counts.get(class_idx, 0) < samples_per_class:
                        layer_class_acts[layer_name][class_idx].append(
                            batch_acts[i].cpu().clone()
                        )

            # Class counts from any non-skipped layer
            for i, label in enumerate(labels):
                class_idx = label.item()
                if samples_per_class is None or class_counts[class_idx] < samples_per_class:
                    class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

            # Stop once every class has samples_per_class samples
            if samples_per_class is not None:
                if all(count >= samples_per_class for count in class_counts.values()):
                    break

            if batch_idx % 10 == 0:
                _clear_memory_cache()

    for hook in hooks:
        hook.remove()

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
                feature_dim = 1
                for other_idx in range(num_classes):
                    other_samples = layer_class_acts[layer_name][other_idx]
                    if other_samples:
                        feature_dim = other_samples[0].shape[0]
                        break
                layer_result[class_idx] = torch.zeros(0, feature_dim)
        if has_data:
            result[layer_name] = layer_result

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
    """Mean activation per class at each tracked layer.

    Args:
        model: Model to extract from
        data_loader: Data loader
        device: Device to run on
        num_classes: Number of classes
        use_disk: If True, stream activations to disk to reduce RAM usage
        temp_dir: Directory for temporary storage (required if use_disk=True)

    Returns:
        {
            layer_name: {class_idx: mean_activation_tensor}
        }
    """
    if use_disk:
        if temp_dir is None:
            raise ValueError("temp_dir required when use_disk=True")
        return extract_multilayer_activations_disk(model, data_loader, device, num_classes, temp_dir)
    was_training = model.training
    model.eval()

    layer_hooks = []
    layer_names = []

    pool_method = _get_pool_method(model)
    if _is_cct(model):  # CCT
        for i in range(len(model.transformer.layers)):
            layer_hooks.append(f'transformer.layers.{i}')
        layer_hooks.append('norm')
    elif hasattr(model, 'fc'):  # ResNet-like architectures
        # Each layer group and avgpool
        for name, module in model.named_modules():
            if 'layer' in name and name.count('.') == 0:
                layer_hooks.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                layer_hooks.append('avgpool')
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
        # After each maxpool in features
        for i, module in enumerate(model.features):
            if isinstance(module, nn.MaxPool2d):
                layer_hooks.append(f'features.{i}')
    elif hasattr(model, 'heads'):  # ViT
        # Each transformer block
        for i, block in enumerate(model.encoder.layers):
            layer_hooks.append(f'encoder.layers.{i}')
        layer_hooks.append('encoder.ln')
    else:  # Other architectures: pooling layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                layer_hooks.append(name)

    activations = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach()
        return hook

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

    layer_class_activations = {layer: {i: [] for i in range(num_classes)} for layer in layer_names}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            for layer_name in layer_names:
                if layer_name in activations:
                    batch_acts = activations[layer_name]

                    batch_acts = _reduce_activation_dims(batch_acts, pool_method)

                    for i, label in enumerate(labels):
                        class_idx = label.item()
                        layer_class_activations[layer_name][class_idx].append(batch_acts[i].cpu())

    for hook in hooks:
        hook.remove()

    result = {}
    for layer_name in layer_names:
        result[layer_name] = {}
        for class_idx in range(num_classes):
            if layer_class_activations[layer_name][class_idx]:
                result[layer_name][class_idx] = torch.stack(
                    layer_class_activations[layer_name][class_idx]
                ).mean(dim=0)
            else:
                # No samples for this class: infer the dimension from a populated class
                feature_dim = 1
                for other_idx in range(num_classes):
                    if layer_class_activations[layer_name][other_idx]:
                        feature_dim = layer_class_activations[layer_name][other_idx][0].shape
                        break
                result[layer_name][class_idx] = torch.zeros(feature_dim)

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
    """Disk-backed version of extract_multilayer_activations.

    Streams activations to disk to limit RAM usage for large models.

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

    was_training = model.training
    model.eval()

    # Same layer selection as extract_multilayer_activations
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

    # Layout: temp_dir/layer_name/class_idx/batch_N.pt
    temp_dir.mkdir(parents=True, exist_ok=True)
    layer_dirs = {}
    for layer in layer_names:
        layer_dir = temp_dir / layer.replace('.', '_')
        layer_dir.mkdir(exist_ok=True)
        layer_dirs[layer] = layer_dir
        for class_idx in range(num_classes):
            (layer_dir / str(class_idx)).mkdir(exist_ok=True)

    batch_counters = {layer: {i: 0 for i in range(num_classes)} for layer in layer_names}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            for layer_name in layer_names:
                if layer_name in activations:
                    batch_acts = activations[layer_name]

                    batch_acts = _reduce_activation_dims(batch_acts, pool_method)

                    for i, label in enumerate(labels):
                        class_idx = label.item()
                        class_acts = batch_acts[i:i+1].cpu().clone()
                        # .clone() detaches from shared storage so torch.save
                        # serialises only the slice, not the full batch tensor
                        batch_num = batch_counters[layer_name][class_idx]
                        save_path = layer_dirs[layer_name] / str(class_idx) / f"batch_{batch_num}.pt"
                        torch.save(class_acts, save_path)
                        batch_counters[layer_name][class_idx] += 1

    for hook in hooks:
        hook.remove()

    result = {}
    for layer_name in layer_names:
        result[layer_name] = {}
        for class_idx in range(num_classes):
            class_dir = layer_dirs[layer_name] / str(class_idx)
            batch_files = sorted(class_dir.glob("batch_*.pt"))

            if batch_files:
                # Running mean over batches, one batch in memory at a time
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
                        # new_mean = old_mean + (batch_mean - old_mean) * batch_size / total_count
                        total_count += batch_size
                        delta = batch_mean - running_mean
                        running_mean += delta * (batch_size / total_count)

                    del batch_acts, batch_mean

                result[layer_name][class_idx] = running_mean
            else:
                # No samples for this class: infer the dimension from the other classes
                feature_dim = 1
                for other_idx in range(num_classes):
                    if result[layer_name].get(other_idx) is not None:
                        feature_dim = result[layer_name][other_idx].shape[0]
                        break
                result[layer_name][class_idx] = torch.zeros(feature_dim)

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp files: {e}")

    if was_training:
        model.train()

    return result


def _detect_probe_layers(model: nn.Module) -> List[str]:
    """Select the layers to probe for the given architecture.

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
    - Other: pooling layers and top-level modules only

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

        if hasattr(model, 'features'):
            for i, stage in enumerate(model.features):
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
            if 'denseblock' in name or 'transition' in name:
                if name.count('.') == 1:  # top-level blocks only
                    probe_layers.append(name)

        if hasattr(model, 'features') and hasattr(model.features, 'norm5'):
            probe_layers.append('features.norm5')

    # EfficientNet / MobileNet
    elif hasattr(model, 'features') and any('inverted' in str(type(m)).lower() for m in model.modules()):
        print("Detected EfficientNet/MobileNet architecture")

        if hasattr(model, 'features'):
            for i, block in enumerate(model.features):
                # every second block
                if i % 2 == 0 or i == len(model.features) - 1:
                    probe_layers.append(f'features.{i}')

        if hasattr(model, 'avgpool'):
            probe_layers.append('avgpool')

    # ConvNeXt
    elif hasattr(model, 'features') and any('convnext' in str(type(m)).lower() for m in model.modules()):
        print("Detected ConvNeXt architecture")

        for name, module in model.named_modules():
            if 'features' in name and name.count('.') == 2:
                probe_layers.append(name)

        if hasattr(model, 'avgpool'):
            probe_layers.append('avgpool')

    # MaxViT
    elif any('maxvit' in str(type(m)).lower() for m in model.modules()):
        print("Detected MaxViT architecture")

        for name, module in model.named_modules():
            if 'stem' in name and name.count('.') <= 1:
                probe_layers.append(name)
            elif 'stages' in name and name.count('.') == 2:
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

    # UNet: detected by the inc, down1-4, up1-4 modules
    elif all(hasattr(model, attr) for attr in ['inc', 'down1', 'down2', 'down3', 'down4']):
        print("Detected UNet architecture")

        probe_layers.extend(['inc', 'down1', 'down2', 'down3', 'down4'])

        if hasattr(model, 'up1'):
            probe_layers.extend(['up1', 'up2', 'up3', 'up4'])

        if hasattr(model, 'global_pool'):
            probe_layers.append('global_pool')

    # Other architectures: pooling layers and top-level modules only
    else:
        print("Using fallback layer detection for custom architecture")
        print("  WARNING: Limiting to depth=1 and pooling layers only to avoid excessive probing")

        for name, module in model.named_modules():
            if name == '':
                continue

            depth = name.count('.')

            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                if depth <= 2:
                    probe_layers.append(name)

            elif depth == 1 and isinstance(module, (
                nn.Sequential,
                nn.ModuleList
            )):
                probe_layers.append(name)

    seen = set()
    probe_layers = [x for x in probe_layers if not (x in seen or seen.add(x))]

    # More than 50 layers: keep the first and last and subsample the rest evenly
    if len(probe_layers) > 50:
        print(f"  WARNING: {len(probe_layers)} layers detected, subsampling to 50 evenly spaced layers")
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
    """Train linear probes on frozen representations from each layer.

    Activations of all probed layers are collected in one pass over the train and
    validation sets; a logistic-regression probe is then fitted per layer.

    Args:
        model: Model to probe
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        num_classes: Number of classes
        use_disk: If True, store activations on disk to reduce RAM usage
        temp_dir: Directory for temporary activation storage (required if use_disk=True)

    Returns:
        {
            'layer_accuracies': {layer_name: {'train': float, 'val': float}},
            'layer_dimensions': {layer_name: int},
            'best_layer': str,
            'best_val_accuracy': float
        }
    """
    if use_disk:
        if temp_dir is None:
            raise ValueError("temp_dir required when use_disk=True")
        return train_linear_probes_disk(model, train_loader, val_loader, device, num_classes, temp_dir)
    was_training = model.training
    model.eval()

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

    # One pass over the training data for all layers
    print("Collecting train activations...")
    train_data = {layer: [] for layer in valid_layers}
    train_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            model(inputs)

            for layer_path in valid_layers:
                if layer_path in activations:
                    train_data[layer_path].append(activations[layer_path].cpu().numpy())

            train_labels.append(labels.numpy())

    train_labels = np.concatenate(train_labels, axis=0)

    # One pass over the validation data for all layers
    print("Collecting validation activations...")
    val_data = {layer: [] for layer in valid_layers}
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            model(inputs)

            for layer_path in valid_layers:
                if layer_path in activations:
                    val_data[layer_path].append(activations[layer_path].cpu().numpy())

            val_labels.append(labels.numpy())

    val_labels = np.concatenate(val_labels, axis=0)

    for hook in hooks:
        hook.remove()

    print("Training linear probes...")
    result = {
        'layer_accuracies': {},
        'layer_dimensions': {},
        'best_layer': None,
        'best_val_accuracy': 0.0
    }

    for idx, layer_path in enumerate(valid_layers):
        print(f"  [{idx+1}/{len(valid_layers)}] Training probe for layer: {layer_path}...", end='', flush=True)

        if not train_data[layer_path] or not val_data[layer_path]:
            print(f" SKIPPED (no activations)")
            train_data.pop(layer_path, None)
            val_data.pop(layer_path, None)
            continue

        X_train = np.concatenate(train_data[layer_path], axis=0)
        del train_data[layer_path]

        X_val = np.concatenate(val_data[layer_path], axis=0)
        del val_data[layer_path]

        # Replace NaN/Inf with 0 (can occur in early-training snapshots with unstable
        # initialisations)
        if not np.isfinite(X_train).all():
            nan_count = (~np.isfinite(X_train)).sum()
            print(f" [warning: {nan_count} non-finite values in train activations, replacing with 0]", end='', flush=True)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(X_val).all():
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardise to avoid overflow in the lbfgs solver
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        # After StandardScaler, std=0 (dead units) gives NaN (set to 0) and std~0 gives
        # very large values (clipped to [-10, 10])
        X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

        probe = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            C=1.0,
            tol=1e-4,
            random_state=42,
            n_jobs=-1
        )

        try:
            probe.fit(X_train, train_labels)

            train_pred = probe.predict(X_train)
            val_pred = probe.predict(X_val)

            train_acc = accuracy_score(train_labels, train_pred) * 100
            val_acc = accuracy_score(val_labels, val_pred) * 100

            result['layer_accuracies'][layer_path] = {
                'train': train_acc,
                'val': val_acc
            }
            result['layer_dimensions'][layer_path] = X_train.shape[1]

            if val_acc > result['best_val_accuracy']:
                result['best_val_accuracy'] = val_acc
                result['best_layer'] = layer_path

            print(f" Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        except Exception as e:
            print(f" FAILED: {e}")

        del X_train, X_val
        _clear_memory_cache()

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
    """Disk-backed version of train_linear_probes.

    Train and validation activations are streamed to per-layer files, then each
    layer's probe is fitted from disk and the temporary files are removed.

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

    was_training = model.training
    model.eval()

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

    temp_dir.mkdir(parents=True, exist_ok=True)
    train_act_dir = temp_dir / "train_activations"
    val_act_dir = temp_dir / "val_activations"
    train_act_dir.mkdir(exist_ok=True)
    val_act_dir.mkdir(exist_ok=True)

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

    # Stream train activations to disk, one .npy file per batch and layer
    print("Collecting train activations (streaming to disk)...")
    train_batch_counts = {layer: 0 for layer in valid_layers}

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            model(inputs)

            for layer_path in valid_layers:
                if layer_path in activations:
                    act_np = activations[layer_path].cpu().numpy()
                    layer_filename = layer_path.replace('.', '_')
                    batch_file = train_act_dir / f"{layer_filename}_batch_{train_batch_counts[layer_path]}.npy"
                    np.save(batch_file, act_np, allow_pickle=False)
                    label_file = train_act_dir / f"{layer_filename}_labels_batch_{train_batch_counts[layer_path]}.npy"
                    np.save(label_file, labels.numpy(), allow_pickle=False)
                    train_batch_counts[layer_path] += 1

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
                    batch_file = val_act_dir / f"{layer_filename}_batch_{val_batch_counts[layer_path]}.npy"
                    np.save(batch_file, act_np, allow_pickle=False)
                    label_file = val_act_dir / f"{layer_filename}_labels_batch_{val_batch_counts[layer_path]}.npy"
                    np.save(label_file, labels.numpy(), allow_pickle=False)
                    val_batch_counts[layer_path] += 1

    for hook in hooks:
        hook.remove()

    # Fit probes layer by layer from disk
    print("Training linear probes from disk-stored activations...")
    result = {
        'layer_accuracies': {},
        'layer_dimensions': {},
        'best_layer': None,
        'best_val_accuracy': 0.0
    }

    for idx, layer_path in enumerate(valid_layers):
        print(f"  [{idx+1}/{len(valid_layers)}] Training probe for {layer_path}...", end='', flush=True)

        layer_filename = layer_path.replace('.', '_')

        try:
            if train_batch_counts[layer_path] == 0 or val_batch_counts[layer_path] == 0:
                print(" SKIPPED (no activations)")
                continue

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
            del train_batches, train_label_batches

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
            del val_batches, val_label_batches

            # Replace NaN/Inf with 0 (can occur in early-training snapshots with unstable
            # initialisations)
            if not np.isfinite(X_train).all():
                nan_count = (~np.isfinite(X_train)).sum()
                print(f" [warning: {nan_count} non-finite values in train activations, replacing with 0]", end='', flush=True)
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(X_val).all():
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

            # Standardise to avoid overflow in the lbfgs solver
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            # After StandardScaler, std=0 (dead units) gives NaN (set to 0) and std~0 gives
            # very large values (clipped to [-10, 10])
            X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
            X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

            probe = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                C=1.0,
                tol=1e-4,
                random_state=42,
                n_jobs=-1
            )

            probe.fit(X_train, y_train)

            train_pred = probe.predict(X_train)
            val_pred = probe.predict(X_val)

            train_acc = accuracy_score(y_train, train_pred) * 100
            val_acc = accuracy_score(y_val, val_pred) * 100

            feature_dim = X_train.shape[1]

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

    try:
        shutil.rmtree(train_act_dir, ignore_errors=True)
        shutil.rmtree(val_act_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp files: {e}")

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
    """Compute per-class performance metrics.

    Args:
        model: Neural network model
        data_loader: DataLoader for evaluation
        device: Device to run on ('cuda', 'cpu', 'mps')
        class_names: List of class names
        num_classes: Number of classes (optional, derived from class_names if not provided)

    Returns:
        List of dictionaries with class metrics
    """
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

    if was_training:
        model.train()

    return metrics
