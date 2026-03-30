"""
Weight initialization methods for neural networks
"""
import torch
import torch.nn as nn
from typing import Literal, Optional

InitializationMethod = Literal[
    "random_normal",
    "random_uniform",
    "xavier_normal",
    "xavier_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "pretrained_imagenet",
    "zeros",
    "ones",
]


def initialize_weights(
    model: nn.Module,
    method: InitializationMethod,
    seed: Optional[int] = None,
) -> None:
    """
    Initialize model weights using the specified method.

    Args:
        model: PyTorch model to initialize
        method: Initialization method to use
        seed: Random seed for reproducibility (optional)

    Note:
        - This function modifies the model in-place
        - 'pretrained_imagenet' should be handled during model creation, not here
        - Biases are typically initialized to zeros except for special cases
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if method == "pretrained_imagenet":
        # Pretrained weights are loaded during model creation
        # This is a no-op, but we include it for completeness
        return

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if method == "random_normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

            elif method == "random_uniform":
                nn.init.uniform_(module.weight, a=-0.05, b=0.05)

            elif method == "xavier_normal":
                nn.init.xavier_normal_(module.weight)

            elif method == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)

            elif method == "kaiming_normal":
                # fan_out: preserves gradient variance through backward pass — the
                # PyTorch default for ResNet. Better than fan_in for deep networks.
                # Use fan_in only for Linear layers (no spatial dims to average over).
                mode = "fan_in" if isinstance(module, nn.Linear) else "fan_out"
                nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity="relu")

            elif method == "kaiming_uniform":
                mode = "fan_in" if isinstance(module, nn.Linear) else "fan_out"
                nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity="relu")

            elif method == "zeros":
                nn.init.zeros_(module.weight)

            elif method == "ones":
                nn.init.ones_(module.weight)

            else:
                raise ValueError(f"Unknown initialization method: {method}")

            # Initialize biases to zero (standard practice)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def save_weights(model: nn.Module, path: str) -> None:
    """
    Save model weights to disk.

    Args:
        model: PyTorch model
        path: File path to save weights (.pt file)
    """
    torch.save(model.state_dict(), path)


def load_weights(model: nn.Module, path: str, device: str = "cpu") -> None:
    """
    Load model weights from disk.

    Args:
        model: PyTorch model to load weights into
        path: File path to load weights from (.pt file)
        device: Device to load weights onto (cpu, cuda, mps)
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
