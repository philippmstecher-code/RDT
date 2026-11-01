"""
Visualization methods for neural network representations.

Activation maximisation, saliency maps, feature-map grids and 2D trajectories of
class representations across training snapshots.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; must be set before importing pyplot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def generate_deep_dream(
    model: nn.Module,
    class_idx: int,
    device: torch.device,
    num_iterations: int = 200,
    learning_rate: float = 0.05,
    image_size: Tuple[int, int] = (224, 224),
    l2_decay: float = 0.0001,
    tv_weight: float = 0.0001,
    blur_every: int = 4,
    clip_percentile: int = 1
) -> np.ndarray:
    """
    Generate optimal input that maximizes activation for a specific class.

    Uses gradient ascent to find an input image that maximally activates the
    target class output. Applies regularization to produce interpretable images.

    Args:
        model: Neural network model in eval mode
        class_idx: Target class index to maximize
        device: Torch device (cpu/cuda)
        num_iterations: Number of optimization steps
        learning_rate: Step size for gradient ascent
        image_size: Output image dimensions (H, W)
        l2_decay: L2 regularization weight
        tv_weight: Total variation regularization weight
        blur_every: Apply Gaussian blur every N iterations
        clip_percentile: Percentile for clipping extreme values

    Returns:
        RGB image as numpy array (H, W, 3) in range [0, 255]
    """
    model.eval()

    img = torch.randn(1, 3, image_size[0], image_size[1], device=device)
    img.requires_grad_(True)

    optimizer = torch.optim.Adam([img], lr=learning_rate)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        output = model(img)

        # Target: maximize logit for target class
        target_activation = output[0, class_idx]

        l2_reg = l2_decay * torch.sum(img ** 2)
        tv_reg = tv_weight * total_variation_loss(img)

        # Minimise the negative logit (gradient ascent on the class score)
        loss = -target_activation + l2_reg + tv_reg

        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            torch.cuda.empty_cache()

        # Apply blur periodically to reduce high-frequency noise
        if iteration % blur_every == 0:
            img.data = gaussian_blur(img.data, kernel_size=3)

        if iteration % 10 == 0:
            img.data = percentile_clip(img.data, clip_percentile)

    img_np = tensor_to_image(img[0])

    return img_np


def generate_gradient_saliency(
    model: nn.Module,
    class_idx: int,
    representative_sample: torch.Tensor,
    device: torch.device,
    overlay_alpha: float = 0.5
) -> np.ndarray:
    """
    Generate gradient-based saliency map for a class.

    Computes gradients of the target class score with respect to the input
    to identify which pixels are most important for the classification.

    Args:
        model: Neural network model in eval mode
        class_idx: Target class index
        representative_sample: Input image tensor (C, H, W)
        device: Torch device
        overlay_alpha: Blending factor for heatmap overlay (0=original, 1=heatmap)

    Returns:
        RGB heatmap overlay as numpy array (H, W, 3)
    """
    model.eval()

    img = representative_sample.unsqueeze(0).to(device)
    img.requires_grad_(True)

    output = model(img)

    # Gradient of the target-class score w.r.t. the input
    model.zero_grad()
    target_score = output[0, class_idx]
    target_score.backward()

    gradients = img.grad[0]  # (C, H, W)

    # Aggregate across channels (take absolute value and mean)
    saliency = torch.abs(gradients).mean(dim=0)  # (H, W)

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    saliency_np = saliency.detach().cpu().numpy()

    heatmap = apply_colormap(saliency_np, cmap='jet')

    original_img = tensor_to_image(representative_sample)

    blended = (1 - overlay_alpha) * original_img + overlay_alpha * heatmap
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def generate_feature_maps(
    model: nn.Module,
    class_idx: int,
    representative_sample: torch.Tensor,
    device: torch.device,
    layer_name: Optional[str] = None,
    top_k: int = 16
) -> np.ndarray:
    """
    Visualize intermediate layer activations for a class.

    Extracts feature maps from a specified layer and visualizes the most
    activated channels in a grid layout.

    Args:
        model: Neural network model
        class_idx: Target class index (for sample selection)
        representative_sample: Input image tensor (C, H, W)
        device: Torch device
        layer_name: Name of layer to visualize (auto-detected if None)
        top_k: Number of top activated channels to visualize

    Returns:
        Grid of feature maps as numpy array (H, W, 3)
    """
    model.eval()

    if layer_name is None:
        layer_name = _detect_visualization_layer(model)

    activations = {}

    def hook_fn(module, input, output):
        activations['features'] = output

    target_layer = _get_layer_by_name(model, layer_name)
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    hook = target_layer.register_forward_hook(hook_fn)

    try:
        img = representative_sample.unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(img)

        features = activations['features'][0]  # (C, H, W)

        # Select top-k most activated channels
        channel_means = features.mean(dim=(1, 2))  # (C,)
        top_indices = torch.topk(channel_means, min(top_k, features.shape[0])).indices

        selected_features = features[top_indices]  # (top_k, H, W)

        grid_img = create_feature_map_grid(selected_features)

    finally:
        hook.remove()

    return grid_img


def generate_projection_evolution(
    snapshots_data: List[Dict[str, any]],
    method: str = 'tsne',
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> np.ndarray:
    """
    Generate 2D projection showing representation space evolution.

    Shows how class representations move through feature space during training
    by projecting mean activations from multiple snapshots into 2D.

    Args:
        snapshots_data: List of dicts with keys:
            - 'milestone_index': int
            - 'epoch': int
            - 'activations': Dict[int, torch.Tensor] (class_idx -> mean activation)
        method: Dimensionality reduction method ('tsne' or 'pca')
        class_names: Optional list of class names for labels
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure as numpy array (H, W, 3)
    """
    all_activations = []
    all_labels = []
    all_snapshot_indices = []

    for snapshot in snapshots_data:
        milestone_idx = snapshot['milestone_index']
        activations_dict = snapshot['activations']

        for class_idx, activation_tensor in activations_dict.items():
            activation = activation_tensor.cpu().numpy()
            if len(activation.shape) > 1:
                activation = activation.flatten()

            all_activations.append(activation)
            all_labels.append(class_idx)
            all_snapshot_indices.append(milestone_idx)

    X = np.stack(all_activations)  # (n_samples, n_features)
    labels = np.array(all_labels)
    snapshot_indices = np.array(all_snapshot_indices)

    # Sanitize: replace NaN/Inf with 0 to avoid sklearn numerical explosions
    # (can occur with early-training activations from near-random weights)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_.sum()
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        X_2d = reducer.fit_transform(X)
        explained_var = None
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=figsize)

    unique_classes = np.unique(labels)
    unique_snapshots = np.unique(snapshot_indices)
    num_snapshots = len(unique_snapshots)

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    # Plot each class's trajectory
    for i, class_idx in enumerate(unique_classes):
        class_mask = labels == class_idx
        class_label = class_names[class_idx] if class_names else f"Class {class_idx}"

        class_points = X_2d[class_mask]
        class_snapshots = snapshot_indices[class_mask]

        sort_idx = np.argsort(class_snapshots)
        class_points = class_points[sort_idx]
        class_snapshots = class_snapshots[sort_idx]

        # Plot trajectory with arrows
        for j in range(len(class_points)):
            # Alpha increases with training progress
            alpha = 0.3 + 0.7 * (class_snapshots[j] / (num_snapshots - 1))

            # Size increases with training progress
            size = 50 + 150 * (class_snapshots[j] / (num_snapshots - 1))

            # First point: mark as start
            if j == 0:
                ax.scatter(class_points[j, 0], class_points[j, 1],
                          c=[colors[i]], s=size, alpha=alpha,
                          marker='o', edgecolors='black', linewidths=1)
            # Last point: mark as end
            elif j == len(class_points) - 1:
                ax.scatter(class_points[j, 0], class_points[j, 1],
                          c=[colors[i]], s=size, alpha=alpha,
                          marker='*', edgecolors='black', linewidths=1,
                          label=class_label)
            else:
                ax.scatter(class_points[j, 0], class_points[j, 1],
                          c=[colors[i]], s=size, alpha=alpha,
                          marker='o', edgecolors='black', linewidths=0.5)

            # Draw arrow to next point
            if j < len(class_points) - 1:
                ax.annotate('', xy=class_points[j+1], xytext=class_points[j],
                           arrowprops=dict(arrowstyle='->', color=colors[i],
                                         alpha=alpha*0.5, lw=1.5))

    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)

    if method == 'pca' and explained_var is not None:
        title = f'Representation Space Evolution (PCA, {explained_var:.1%} variance)'
    else:
        title = 'Representation Space Evolution (t-SNE)'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    img_array = fig_to_array(fig)
    plt.close(fig)

    return img_array


# Helper Functions

def load_representative_samples(
    dataloader,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 5
) -> Dict[int, torch.Tensor]:
    """
    Load representative samples for each class from a dataloader.

    Args:
        dataloader: PyTorch DataLoader
        num_classes: Number of classes
        device: Torch device
        samples_per_class: Number of samples to collect per class

    Returns:
        Dictionary mapping class_idx -> tensor of samples (N, C, H, W)
    """
    samples = {i: [] for i in range(num_classes)}

    for batch_data, batch_labels in dataloader:
        for i in range(len(batch_labels)):
            label = batch_labels[i].item()
            if label < num_classes and len(samples[label]) < samples_per_class:
                samples[label].append(batch_data[i])

        if all(len(samples[i]) >= samples_per_class for i in range(num_classes)):
            break

    for label in samples:
        if len(samples[label]) > 0:
            samples[label] = torch.stack(samples[label]).to(device)
        else:
            # Classes without samples get a random-noise placeholder
            samples[label] = torch.randn(1, 3, 224, 224).to(device)

    return samples


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor to displayable image array.

    Args:
        tensor: Image tensor (C, H, W) in range ~[-2, 2] (ImageNet normalized)

    Returns:
        RGB image array (H, W, 3) in range [0, 255]
    """
    # Denormalize (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    if tensor.device.type == 'cuda':
        mean = mean.cuda()
        std = std.cuda()

    img = tensor * std + mean

    img = torch.clamp(img, 0, 1)

    # Convert to numpy and transpose to (H, W, C)
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0)

    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def apply_colormap(values: np.ndarray, cmap: str = 'jet') -> np.ndarray:
    """
    Apply colormap to grayscale values.

    Args:
        values: 2D array of values in range [0, 1]
        cmap: Matplotlib colormap name

    Returns:
        RGB image array (H, W, 3) in range [0, 255]
    """
    colormap = cm.get_cmap(cmap)
    colored = colormap(values)  # Returns (H, W, 4) with RGBA
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return rgb


def create_feature_map_grid(features: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create grid visualization of feature maps.

    Args:
        features: Feature maps tensor (C, H, W)
        grid_size: Optional (rows, cols) for grid layout

    Returns:
        Grid image as numpy array (H, W, 3)
    """
    num_features = features.shape[0]

    if grid_size is None:
        grid_cols = int(np.ceil(np.sqrt(num_features)))
        grid_rows = int(np.ceil(num_features / grid_cols))
    else:
        grid_rows, grid_cols = grid_size

    # Normalize each feature map to [0, 1]
    features_normalized = []
    for i in range(num_features):
        feat = features[i].cpu().numpy()
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        features_normalized.append(feat)

    h, w = features.shape[1], features.shape[2]
    grid = np.zeros((grid_rows * h, grid_cols * w))

    for idx, feat in enumerate(features_normalized):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = feat

    grid_colored = apply_colormap(grid, cmap='viridis')

    return grid_colored


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation loss for regularization.

    Encourages spatial smoothness by penalizing differences between
    adjacent pixels.

    Args:
        img: Image tensor (B, C, H, W)

    Returns:
        Scalar loss value
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def gaussian_blur(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Apply Gaussian blur to image tensor.

    Args:
        img: Image tensor (B, C, H, W)
        kernel_size: Size of Gaussian kernel (must be odd)

    Returns:
        Blurred image tensor
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    padding = kernel_size // 2

    sigma = kernel_size / 6.0
    x = torch.arange(-padding, padding + 1, dtype=torch.float32, device=img.device)
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # 2D kernel
    kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

    kernel_2d = kernel_2d.expand(img.shape[1], 1, kernel_size, kernel_size)

    blurred = F.conv2d(img, kernel_2d, padding=padding, groups=img.shape[1])

    return blurred


def percentile_clip(img: torch.Tensor, percentile: int = 1) -> torch.Tensor:
    """
    Clip extreme values using percentiles.

    Args:
        img: Image tensor
        percentile: Percentile for clipping (e.g., 1 means clip bottom 1% and top 1%)

    Returns:
        Clipped image tensor
    """
    low = torch.quantile(img, percentile / 100.0)
    high = torch.quantile(img, 1.0 - percentile / 100.0)
    return torch.clamp(img, low, high)


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """
    Convert matplotlib figure to numpy array.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB image array (H, W, 3)
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)[:, :, :3]  # Remove alpha channel if present
    buf.close()
    return img_array


def _detect_visualization_layer(model: nn.Module) -> str:
    """
    Auto-detect appropriate layer for feature visualization.

    Args:
        model: Neural network model

    Returns:
        Layer name as string
    """
    # Common patterns for different architectures
    named_modules = dict(model.named_modules())

    # ResNet
    if 'layer4' in named_modules:
        return 'layer4'

    # VGG
    if 'features' in named_modules:
        return 'features'

    # EfficientNet
    if 'features.7' in named_modules:
        return 'features.7'

    # ViT
    if 'encoder.ln' in named_modules:
        return 'encoder.ln'

    # Default: last conv layer
    last_conv = None
    for name, module in named_modules.items():
        if isinstance(module, nn.Conv2d):
            last_conv = name

    if last_conv:
        return last_conv

    raise ValueError("Could not auto-detect visualization layer")


def _get_layer_by_name(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """
    Get layer from model by name.

    Args:
        model: Neural network model
        layer_name: Name of layer (e.g., 'layer4')

    Returns:
        Layer module or None if not found
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def save_visualization(
    img_array: np.ndarray,
    save_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save visualization image to disk.

    Args:
        img_array: RGB image array (H, W, 3)
        save_path: Path to save PNG file
        metadata: Optional metadata to save alongside
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(img_array)
    img.save(save_path)

    if metadata:
        import json
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
