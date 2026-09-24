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

It can also lie about its dataset size (`claimed_num_examples`). Plain
FedAvg weights each update by the example count the client reports, so an
attacker reporting its true, possibly small, count barely moves the
average; attackers in the poisoning literature inflate it. The trust
filter and trimmed mean ignore reported counts, so this only strengthens
the attack against the FedAvg baseline -- which is the comparison it's for.
"""

from __future__ import annotations

from flwr.common import NDArrays, Scalar

from fl_ids.fl.client import AutoencoderClient
from fl_ids.utils.config import RobustnessConfig

ATTACKER_COUNT_MODES = ("honest", "max_client")


def attacker_claimed_examples(config: RobustnessConfig, client_train_sizes: list[int]) -> int | None:
    """The example count attackers report, per `robustness.attacker_example_count`.

    Args:
        config: Robustness config.
        client_train_sizes: Every client's training-row count (the harness
            running the federation knows these; a real attacker would just
            claim a large number).

    Returns:
        None for "honest" (report the true count), else the largest
        client's training-row count.

    Raises:
        ValueError: On an unknown mode.
    """
    mode = config.attacker_example_count
    if mode == "honest":
        return None
    if mode == "max_client":
        return int(max(client_train_sizes))
    raise ValueError(f"robustness.attacker_example_count must be one of {ATTACKER_COUNT_MODES}, got {mode!r}")


class SignFlipAttackerClient(AutoencoderClient):
    """Trains normally, then negates and amplifies its own true delta
    before returning it — corrupting the *update*, not the local data.
    """

    def __init__(
        self, *args, amplification: float = 5.0, claimed_num_examples: int | None = None, **kwargs
    ) -> None:
        """Initialize the attacker.

        Args:
            *args: Forwarded to `AutoencoderClient.__init__`.
            amplification: Magnitude multiplier applied to the negated
                true delta (component 5's "negate and amplify").
            claimed_num_examples: Example count to report instead of the
                true one (see module docstring); None reports honestly.
            **kwargs: Forwarded to `AutoencoderClient.__init__`.
        """
        super().__init__(*args, **kwargs)
        self.amplification = amplification
        self.claimed_num_examples = claimed_num_examples

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train honestly, then return a negated, amplified version of the true delta (and the claimed count)."""
        honest_weights, num_examples, metrics = super().fit(parameters, config)
        corrupted_weights = [
            received - self.amplification * (trained - received)
            for received, trained in zip(parameters, honest_weights)
        ]
        if self.claimed_num_examples is not None:
            num_examples = self.claimed_num_examples
        return corrupted_weights, num_examples, metrics
