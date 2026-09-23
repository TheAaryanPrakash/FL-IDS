"""Client-side autoencoder — cascade stage 2, unsupervised backstop (component 3).

A dense (not convolutional — this is tabular flow data) encoder-decoder
with a small bottleneck, trained with MSE reconstruction loss on whatever
subset of local traffic the caller provides — per CLAUDE.md, that subset
must be the boosting-filtered "normal" traffic (see `fl_ids.fl.client`),
never a client's raw unfiltered local data; this module itself is
data-source-agnostic and just trains on whatever `X` it's given.

**Design note — feature scale:** the boosting classifier (component 2) is
trained on features standardized the same way every client standardizes
its own local data (see `fl_ids.models.boosting`'s module docstring for
the full reasoning) — this module inherits that assumption: `X` passed to
`train_autoencoder`/`reconstruction_error` is expected to already be
per-client-normalized (component 1's output), not raw-scale.

The anomaly threshold is a high percentile of reconstruction error on the
client's benign validation slice, recomputed every round (never
calibrated once and frozen), since the global autoencoder keeps changing
as FL rounds progress.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from fl_ids.utils.config import AutoencoderConfig

logger = logging.getLogger(__name__)


def _build_mlp(dims: list[int]) -> nn.Sequential:
    """Build a dense MLP: ReLU between layers, linear on the final output."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Dense encoder-decoder for tabular flow features.

    Architecture: input -> hidden_dims -> bottleneck_dim (encoder), then
    the mirror image back out to input_dim (decoder).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> None:
        """Initialize the network.

        Args:
            input_dim: Number of input features.
            hidden_dims: Encoder hidden layer widths, in order from input
                towards the bottleneck (decoder mirrors this in reverse).
            bottleneck_dim: Width of the bottleneck layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.encoder = _build_mlp([input_dim, *hidden_dims, bottleneck_dim])
        self.decoder = _build_mlp([bottleneck_dim, *reversed(hidden_dims), input_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input: encode to the bottleneck, then decode back out."""
        return self.decoder(self.encoder(x))


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of numpy arrays (Flower's `NDArrays`)."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    """Load a list of numpy arrays (Flower's `NDArrays`) into a model in place."""
    state_dict = model.state_dict()
    new_state = {
        key: torch.tensor(value, dtype=param.dtype)
        for (key, param), value in zip(state_dict.items(), weights)
    }
    model.load_state_dict(new_state, strict=True)


def train_autoencoder(
    model: Autoencoder,
    X: np.ndarray,
    config: AutoencoderConfig,
    seed: int,
) -> list[float]:
    """Train the autoencoder with MSE reconstruction loss.

    Args:
        model: The autoencoder to train, in place.
        X: Training features (already boosting-filtered by the caller),
            shape (n_samples, input_dim).
        config: Autoencoder hyperparameters (learning_rate, local_epochs,
            batch_size).
        seed: Random seed, for reproducible batch shuffling.

    Returns:
        Mean MSE loss per epoch, in training order.
    """
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    n = len(X_tensor)

    epoch_losses: list[float] = []
    for _ in range(config.local_epochs):
        perm = torch.randperm(n)
        batch_losses = []
        for start in range(0, n, config.batch_size):
            batch = X_tensor[perm[start : start + config.batch_size]]
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
    return epoch_losses


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-sample MSE reconstruction error (mean squared error across features).

    Args:
        model: A trained (or in-training) autoencoder.
        X: Features to evaluate, shape (n_samples, input_dim).

    Returns:
        Per-sample reconstruction error, shape (n_samples,). Empty array if
        `X` is empty.
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstruction = model(X_tensor)
        errors = torch.mean((reconstruction - X_tensor) ** 2, dim=1)
    return errors.numpy()


def compute_anomaly_threshold(errors: np.ndarray, percentile: float) -> float:
    """Anomaly threshold: a high percentile of benign reconstruction error.

    Recomputed every round (per component 3) — never calibrated once and
    frozen, since the global autoencoder keeps changing.

    Args:
        errors: Reconstruction errors on the client's benign validation
            slice.
        percentile: Percentile in [0, 100] (component default: 97th).

    Returns:
        The threshold value, or `inf` if `errors` is empty (no benign
        validation data to calibrate on — nothing can be safely flagged
        anomalous without a real threshold).
    """
    if len(errors) == 0:
        logger.warning("No benign validation errors to calibrate an anomaly threshold from")
        return float("inf")
    return float(np.percentile(errors, percentile))


def reconstruction_error_histogram(
    errors: np.ndarray, bins: int, value_range: tuple[float, float]
) -> np.ndarray:
    """Fixed-bin, fixed-range histogram of reconstruction errors.

    Fixed bins/range (rather than data-dependent bin edges) so histograms
    from different clients/rounds are directly comparable and safely
    aggregatable server-side, per component 4.

    Args:
        errors: Reconstruction errors to bin.
        bins: Number of histogram bins.
        value_range: (min, max) range covered by the histogram.

    Returns:
        Bin counts, shape (bins,), dtype int64.
    """
    counts, _ = np.histogram(errors, bins=bins, range=value_range)
    return counts.astype(np.int64)
