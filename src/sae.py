"""
Sparse Autoencoder (SAE) implementation for feature decomposition.

Provides a configurable SAE with two sparsity modes:
  - L1 penalty: standard ReLU + L1 regularisation on hidden activations
  - k-sparse: top-k activation — zero out all but the k largest hidden units,
    no L1 needed.  Introduced by Gao et al. 2024 ("Scaling and Evaluating
    Sparse Autoencoders"), who showed top-k simplifies hyperparameter tuning
    and improves the reconstruction–sparsity Pareto frontier vs. ReLU + L1.

The decoder columns are normalised to unit norm during training to prevent
trivial solutions (standard practice; Bricken et al. 2023, Gao et al. 2024).

Literature references:
  - Bricken et al. 2023, "Towards Monosemanticity" — SAE architecture baseline
  - Gao et al. 2024, "Scaling and Evaluating Sparse Autoencoders" — top-k SAEs
  - Gorton et al. 2025, "Sparse Autoencoders for Vision Models" — vision SAEs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder with optional top-k sparsity.

    Args:
        d_input: Dimensionality of the input activations.
        expansion_factor: Hidden layer size = d_input * expansion_factor.
        k_sparse: If set, use top-k activation instead of ReLU + L1.
    """

    def __init__(
        self,
        d_input: int,
        expansion_factor: int = 4,
        k_sparse: Optional[int] = None,
    ):
        super().__init__()
        # expansion_factor=4 is on the conservative end.  Literature typically
        # uses 8x–256x for LLMs (Bricken et al. 2023; Cunningham et al. 2023)
        # and 32x for vision (Gorton et al. 2025).  We use 4x as a pragmatic
        # compute-constrained choice since we train one SAE per layer per
        # checkpoint (many SAEs total).
        d_hidden = d_input * expansion_factor
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k_sparse = k_sparse

        self.encoder = nn.Linear(d_input, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_input, bias=True)

        # Initialise decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs to (possibly sparse) hidden activations."""
        h = self.encoder(x)
        if self.k_sparse is not None:
            # Top-k: keep only the k largest activations
            topk_vals, topk_idx = h.topk(self.k_sparse, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(-1, topk_idx, 1.0)
            h = h * mask
        else:
            h = F.relu(h)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden activations back to input space."""
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            (reconstruction, hidden_activations)
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


def _normalize_decoder(model: SparseAutoencoder) -> None:
    """Project decoder weight columns back to unit norm (in-place)."""
    with torch.no_grad():
        norms = model.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        model.decoder.weight.div_(norms)


def create_sae_init(
    d_input: int,
    expansion_factor: int = 4,
    k_sparse: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Create a reproducible SAE initialization state dict.

    Use this to create one init per layer, then pass it to train_sae()
    via init_state so that SAEs at different milestones start from the
    same weights (enabling meaningful decoder-direction matching).
    """
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model = SparseAutoencoder(d_input=d_input, expansion_factor=expansion_factor, k_sparse=k_sparse)
    state = model.state_dict()
    torch.random.set_rng_state(rng_state)
    return state


def train_sae(
    activations: torch.Tensor,
    expansion_factor: int = 4,
    k_sparse: int = 32,
    n_steps: int = 10_000,
    lr: float = 1e-3,
    l1_coeff: float = 1e-3,
    batch_size: int = 256,
    device: Optional[str] = None,
    init_state: Optional[dict] = None,
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on cached activation vectors.

    Default hyperparameters and literature context:

      k_sparse=32:  Directly from Gao et al. 2024, who used k=32 throughout
                    their ablation studies (Figs 15–18, 128k latents on GPT-2).
                    Also used by Inaba et al. 2025 ("How LLMs Learn").

      expansion_factor=4:  Conservative vs. literature (8x–256x for LLMs,
                    32x for vision).  Pragmatic for per-checkpoint training.

      n_steps=10_000:  No direct literature precedent.  Major SAE papers
                    train on billions of activations, but our per-checkpoint
                    activation sets are orders of magnitude smaller.  Bai et
                    al. 2024 ("SAE-Track") showed recurrent initialisation
                    needs ~1/20th of cold-start training budget, providing
                    indirect support for shorter training with shared init.

      l1_coeff=1e-3:  Within the range tested for vision SAEs by Gorton et
                    al. 2025 (4e-4 – 1.6e-3) and at the lower end of Gao
                    et al. 2024 (1.7e-3 – 1.7e-2).  Only used when k_sparse
                    is disabled (ReLU + L1 mode).

      init_state (shared init):  Motivated by Bau et al. 2025, who showed
                    SAEs with different random seeds share only ~30% of
                    learned features.  Shared initialisation ensures decoder
                    directions start comparable across checkpoints.  Related
                    to (but distinct from) Bai et al. 2024's recurrent init,
                    which reuses *trained* weights from the prior checkpoint.

    Args:
        activations: Tensor of shape (N, d_input) — all activation vectors.
        expansion_factor: Hidden layer multiplier.
        k_sparse: If > 0, use top-k sparsity (disable L1).
                  If 0 or None, use ReLU + L1 sparsity.
        n_steps: Number of gradient steps.
        lr: Learning rate.
        l1_coeff: L1 penalty coefficient (ignored when using k-sparse).
        batch_size: Mini-batch size.
        device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
        init_state: Optional state dict from create_sae_init(). If provided,
                    the model loads these weights before training (shared init).

    Returns:
        Trained SparseAutoencoder (moved to CPU).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    activations = activations.float()
    n_samples, d_input = activations.shape
    use_topk = k_sparse is not None and k_sparse > 0

    model = SparseAutoencoder(
        d_input=d_input,
        expansion_factor=expansion_factor,
        k_sparse=k_sparse if use_topk else None,
    )
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        # Sample mini-batch
        idx = torch.randint(0, n_samples, (min(batch_size, n_samples),))
        batch = activations[idx].to(device)

        x_hat, h = model(batch)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, batch)

        # Sparsity penalty
        if use_topk:
            loss = recon_loss
        else:
            l1_loss = h.abs().mean()
            loss = recon_loss + l1_coeff * l1_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Re-normalise decoder columns every step
        _normalize_decoder(model)

    model.eval()
    return model.cpu()
