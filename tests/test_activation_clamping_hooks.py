"""
Unit tests for activation clamping hooks (ClampingHook and MeasurementHook).

Verifies that hooks correctly handle:
  - 2D tensors (B, C) — e.g., after avgpool
  - 4D tensors (B, C, H, W) — CNN layers with spatial dimensions
  - The subtraction-based clamping strategy for spatial layers
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Import from runpodScripts
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runpodScripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sae import SparseAutoencoder


# Re-implement hooks here to test them in isolation (avoiding full import chain)
# This mirrors the hooks in run_activation_clamping.py
from run_activation_clamping import ClampingHook, MeasurementHook


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sae_64():
    """SAE with d_input=64, expansion_factor=4, k_sparse=16."""
    sae = SparseAutoencoder(d_input=64, expansion_factor=4, k_sparse=16)
    sae.eval()
    return sae


class SimpleCNN(nn.Module):
    """Minimal CNN for testing hooks on spatial layers."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 64, 3, padding=1)
        self.layer2 = nn.Conv2d(64, 64, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════
# ClampingHook tests
# ═══════════════════════════════════════════════════════════════════════════

def test_clamping_hook_2d_modifies_output(sae_64):
    """ClampingHook should modify 2D (B, C) output by subtracting Ab-H contribution."""
    hook = ClampingHook(sae_64, zero_indices=[0, 1, 2])
    x = torch.randn(8, 64)
    result = hook(None, None, x)
    # Should be different from input (Ab-H contribution subtracted)
    assert not torch.allclose(result, x, atol=1e-6), \
        "ClampingHook should modify 2D output"
    assert result.shape == x.shape


def test_clamping_hook_4d_modifies_output(sae_64):
    """ClampingHook should modify 4D (B, C, H, W) output — not silently skip."""
    hook = ClampingHook(sae_64, zero_indices=[0, 1, 2])
    x = torch.randn(4, 64, 8, 8)
    result = hook(None, None, x)
    assert result.shape == x.shape, "Output shape should be preserved"
    assert not torch.allclose(result, x, atol=1e-6), \
        "ClampingHook should modify 4D spatial output (was silently skipping before fix)"


def test_clamping_hook_4d_wrong_channels_skips(sae_64):
    """ClampingHook should skip if channel dim doesn't match SAE d_input."""
    hook = ClampingHook(sae_64, zero_indices=[0, 1])
    x = torch.randn(4, 128, 8, 8)  # 128 channels != 64 d_input
    result = hook(None, None, x)
    assert torch.equal(result, x), "Should return original when dims don't match"


def test_clamping_hook_empty_indices_is_identity(sae_64):
    """ClampingHook with no zero_indices should be identity."""
    hook = ClampingHook(sae_64, zero_indices=[])
    x = torch.randn(4, 64, 8, 8)
    result = hook(None, None, x)
    assert torch.equal(result, x)


def test_clamping_hook_4d_preserves_spatial_structure(sae_64):
    """Clamping on 4D should subtract uniformly — spatial structure should be preserved."""
    hook = ClampingHook(sae_64, zero_indices=[0, 1, 2, 3])
    x = torch.randn(2, 64, 4, 4)
    result = hook(None, None, x)
    # The difference should be constant across spatial positions (broadcast subtraction)
    diff = x - result  # (B, C, H, W)
    # For each sample, diff[:, :, i, j] should be the same for all i, j
    for b in range(2):
        spatial_diffs = diff[b]  # (C, H, W)
        ref = spatial_diffs[:, 0, 0]  # (C,) reference
        for i in range(4):
            for j in range(4):
                torch.testing.assert_close(
                    spatial_diffs[:, i, j], ref,
                    msg=f"Spatial position ({i},{j}) should have same diff as (0,0)"
                )


def test_clamping_hook_no_nan_inf(sae_64):
    """Clamping should never produce NaN or Inf."""
    hook = ClampingHook(sae_64, zero_indices=list(range(32)))
    for shape in [(8, 64), (4, 64, 8, 8), (4, 64, 1, 1)]:
        x = torch.randn(*shape)
        result = hook(None, None, x)
        assert not torch.isnan(result).any(), f"NaN in output for shape {shape}"
        assert not torch.isinf(result).any(), f"Inf in output for shape {shape}"


# ═══════════════════════════════════════════════════════════════════════════
# MeasurementHook tests
# ═══════════════════════════════════════════════════════════════════════════

def test_measurement_hook_2d_records_activations(sae_64):
    """MeasurementHook should record activations for 2D input."""
    hook = MeasurementHook(sae_64)
    x = torch.randn(8, 64)
    hook(None, None, x)
    assert hook.activations is not None, "Should record activations for 2D input"
    assert hook.activations.shape == (8, 64 * 4)  # d_hidden = d_input * expansion


def test_measurement_hook_4d_records_activations(sae_64):
    """MeasurementHook should record activations for 4D spatial input — not return None."""
    hook = MeasurementHook(sae_64)
    x = torch.randn(4, 64, 8, 8)
    hook(None, None, x)
    assert hook.activations is not None, \
        "MeasurementHook should record activations for 4D input (was returning None before fix)"
    assert hook.activations.shape == (4, 64 * 4)  # pooled to (B, C) then encoded


def test_measurement_hook_4d_wrong_channels_skips(sae_64):
    """MeasurementHook should return None activations when channels don't match."""
    hook = MeasurementHook(sae_64)
    x = torch.randn(4, 128, 8, 8)
    hook(None, None, x)
    assert hook.activations is None


def test_measurement_hook_passthrough(sae_64):
    """MeasurementHook should not modify the output (pass-through)."""
    hook = MeasurementHook(sae_64)
    x = torch.randn(4, 64, 8, 8)
    result = hook(None, None, x)
    assert torch.equal(result, x), "MeasurementHook should not modify output"


def test_measurement_hook_reset(sae_64):
    """MeasurementHook.reset() should clear stored activations."""
    hook = MeasurementHook(sae_64)
    x = torch.randn(4, 64)
    hook(None, None, x)
    assert hook.activations is not None
    hook.reset()
    assert hook.activations is None


# ═══════════════════════════════════════════════════════════════════════════
# Integration: hooks on actual model forward pass
# ═══════════════════════════════════════════════════════════════════════════

def test_hooks_on_model_forward(sae_64):
    """Test hooks installed on a real model's forward pass with spatial layers."""
    model = SimpleCNN()
    model.eval()

    # Install clamping hook on layer2 (spatial output)
    clamp_hook = ClampingHook(sae_64, zero_indices=[0, 1, 2, 3])
    measure_hook = MeasurementHook(sae_64)

    h1 = model.layer2.register_forward_hook(clamp_hook)
    h2 = model.avgpool.register_forward_hook(measure_hook)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)

    h1.remove()
    h2.remove()

    # Measurement hook on avgpool should have recorded activations
    # (avgpool output is (B, 64, 1, 1) → C=64 matches d_input)
    assert measure_hook.activations is not None, \
        "MeasurementHook on avgpool should record activations"

    # Run again without clamping to verify it made a difference
    measure_hook2 = MeasurementHook(sae_64)
    h3 = model.avgpool.register_forward_hook(measure_hook2)
    with torch.no_grad():
        out_baseline = model(x)
    h3.remove()

    # The clamped run and baseline run should produce different outputs
    # (since we're modifying layer2's output)
    assert not torch.allclose(out, out_baseline, atol=1e-5), \
        "Clamping layer2 should change the model output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
