"""Tests for the sign-flip test attacker's reported example count (component 5's validation attacker)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fl_ids.data.pipeline import partition_and_normalize_clients
from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.simulation import run_simulated_fl_training
from fl_ids.models.autoencoder import Autoencoder, get_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.robustness.attackers import SignFlipAttackerClient, attacker_claimed_examples
from tests.test_strategy import _full_config


def test_claimed_count_modes():
    config = _full_config().robustness
    assert attacker_claimed_examples(dataclasses.replace(config, attacker_example_count="honest"), [5, 90, 20]) is None
    assert attacker_claimed_examples(dataclasses.replace(config, attacker_example_count="max_client"), [5, 90, 20]) == 90
    with pytest.raises(ValueError):
        attacker_claimed_examples(dataclasses.replace(config, attacker_example_count="lots"), [5])


@pytest.fixture(scope="module")
def federation():
    config = _full_config()
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, num_clients=6, dirichlet_alpha=0.3),
        boosting=dataclasses.replace(config.boosting, update_every_n_rounds=0),
    )
    X, y, class_names = make_synthetic_attack_dataset(n_samples=4000, n_features=10, seed=8)
    boosting = BoostingClassifier(config.boosting, len(class_names), 0, 0, config.cascade.confidence_threshold)
    boosting.train(X, y)
    client_data = partition_and_normalize_clients(X, y, 0, config.data, seed=8)
    return config, client_data, boosting, class_names


def test_attacker_reports_the_claimed_count(federation):
    config, client_data, boosting, class_names = federation
    data = client_data[0]
    attacker = SignFlipAttackerClient(
        0, data["X"], data["X_raw"], data["X_val_benign"], config, data["X"].shape[1], claimed_num_examples=12345
    )
    weights = get_weights(Autoencoder(data["X"].shape[1], config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim))
    fit_config = {"boosting_model_bytes": boosting.to_bytes(), "num_classes": len(class_names), "benign_class": 0}
    _, reported, _ = attacker.fit(weights, fit_config)
    assert reported == 12345


def test_inflated_counts_make_attackers_far_more_damaging_under_fedavg(federation):
    config, client_data, boosting, class_names = federation
    # The smallest shard's owner is the attacker: honest reporting gives it
    # little FedAvg weight, inflation gives it as much as the largest client.
    attacker = min(client_data, key=lambda cid: len(client_data[cid]["X"]))

    def final_loss(mode: str) -> float:
        cfg = dataclasses.replace(config, robustness=dataclasses.replace(config.robustness, attacker_example_count=mode))
        result = run_simulated_fl_training(
            client_data, boosting, len(class_names), 0, cfg, num_rounds=4, aggregation="fedavg",
            malicious_client_ids={attacker}, seed=8,
        )
        return result.rounds[-1].mean_val_loss

    honest, inflated = final_loss("honest"), final_loss("max_client")
    assert inflated > 2 * honest, (honest, inflated)
