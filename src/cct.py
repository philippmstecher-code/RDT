"""
Compact Convolutional Transformer (CCT) for small-image classification.

Based on Hassani et al. 2021 — "Escaping the Big Data Paradigm with Compact Transformers".
CCT-7/3×1: 1 conv tokenizer layer (3×3 kernel), 7 transformer layers, attention pooling.
Designed for CIFAR-scale (32×32) images without requiring upscaling.
"""
import torch
import torch.nn as nn


class CCTTokenizer(nn.Module):
    """Convolutional tokenizer: converts images to token sequences."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        n_conv_layers: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
    ):
        super().__init__()
        layers = []
        for i in range(n_conv_layers):
            ic = in_channels if i == 0 else embed_dim
            layers.extend([
                nn.Conv2d(ic, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding),
            ])
        self.tokenizer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)  # [B, embed_dim, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, num_tokens, embed_dim]


class CCT(nn.Module):
    """
    Compact Convolutional Transformer.

    Architecture markers for hook detection:
    - self.transformer (TransformerEncoder with .layers ModuleList)
    - self.norm (LayerNorm)
    - self.fc (final classifier)
    - self._is_cct = True (explicit flag for activation extraction)
    - self._activation_pool = 'mean' (no CLS token — use mean pooling)
    """

    def __init__(
        self,
        num_classes: int = 100,
        embed_dim: int = 256,
        n_conv_layers: int = 1,
        n_transformer_layers: int = 7,
        n_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._is_cct = True
        self._activation_pool = "mean"

        self.tokenizer = CCTTokenizer(embed_dim=embed_dim, n_conv_layers=n_conv_layers)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Attention pooling (no CLS token)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)       # [B, N, D]
        tokens = self.transformer(tokens) # [B, N, D]
        tokens = self.norm(tokens)        # [B, N, D]
        # Attention-weighted sequence pooling
        attn_weights = torch.softmax(self.attention_pool(tokens), dim=1)  # [B, N, 1]
        pooled = (tokens * attn_weights).sum(dim=1)  # [B, D]
        return self.fc(pooled)
