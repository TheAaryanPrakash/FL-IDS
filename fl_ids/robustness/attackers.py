"""Test attacker for the trust filter (component 5's validation case).

CLAUDE.md is explicit about which attack actually validates this defense:
corrupting a malicious client's *local input data* mostly just looks like
non-IID heterogeneity to an unsupervised autoencoder — it never sees
labels, so label-flipping does nothing, and scaled/shifted input just
produces "yet another differently-distributed client." To actually test
the trust filter, the attacker needs to corrupt the *update itself*. A
sign-flip / gradient-ascent attack (negate and amplify the true delta
before sending it) is the standard Byzantine test case this component is
built for.
"""

from __future__ import annotations

from flwr.common import NDArrays, Scalar

from fl_ids.fl.client import AutoencoderClient


class SignFlipAttackerClient(AutoencoderClient):
    """Trains normally, then negates and amplifies its own true delta
    before returning it — corrupting the *update*, not the local data.
    """

    def __init__(self, *args, amplification: float = 5.0, **kwargs) -> None:
        """Initialize the attacker.

        Args:
            *args: Forwarded to `AutoencoderClient.__init__`.
            amplification: Magnitude multiplier applied to the negated
                true delta (component 5's "negate and amplify").
            **kwargs: Forwarded to `AutoencoderClient.__init__`.
        """
        super().__init__(*args, **kwargs)
        self.amplification = amplification

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train honestly, then return a negated, amplified version of the true delta."""
        honest_weights, num_examples, metrics = super().fit(parameters, config)
        corrupted_weights = [
            received - self.amplification * (trained - received)
            for received, trained in zip(parameters, honest_weights)
        ]
        return corrupted_weights, num_examples, metrics
