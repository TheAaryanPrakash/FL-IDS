"""Tests for the incremental boosting update from analyst-confirmed alerts (component 2).

Asserts on what the update does: boosting learns a class it had never
seen once alerts carrying it arrive, only surviving clients' alerts
count, clients surface exactly the rows their autoencoder flags among the
traffic boosting passed, and the real strategy and simulation both run
the update and broadcast the new version afterwards.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from fl_ids.data.pipeline import partition_and_normalize_clients
from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.simulation import run_simulated_fl_training
from fl_ids.fl.client import AutoencoderClient
from fl_ids.fl.strategy import TrustFilteredStrategy
from fl_ids.models.autoencoder import (
    Autoencoder,
    compute_anomaly_threshold,
    get_weights,
    reconstruction_error,
    set_weights,
)
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.models.boosting_update import (
    ALERT_COUNT_KEY,
    SURFACE_ALERTS_KEY,
    BoostingUpdater,
    SurfacedAlerts,
    decode_alerts,
    encode_alerts,
    select_alerts,
)
from tests.test_strategy import _FakeClientProxy, _full_config

UNSEEN = 3  # a class left out of the bootstrap model's training data


@pytest.fixture(scope="module")
def data():
    X, y, class_names = make_synthetic_attack_dataset(n_samples=4000, n_features=10, seed=3)
    return X, y, class_names


@pytest.fixture(scope="module")
def config():
    base = _full_config()
    return dataclasses.replace(
        base,
        boosting=dataclasses.replace(
            base.boosting, num_boost_round=40, update_every_n_rounds=2, update_num_boost_round=40,
            alert_budget_per_client=500,
        ),
    )


@pytest.fixture(scope="module")
def bootstrap(data, config):
    X, y, class_names = data
    seen = y != UNSEEN
    model = BoostingClassifier(config.boosting, len(class_names), 0, 0, config.cascade.confidence_threshold)
    model.train(X[seen], y[seen])
    return model


def test_continued_training_adds_trees_without_touching_the_broadcast_model(data, config, bootstrap):
    X, y, _ = data
    seen = y != UNSEEN
    before = bootstrap.predict_proba(X[:50])

    updated = bootstrap.continue_training(X[seen], y[seen], num_boost_round=40)

    assert updated._booster.current_iteration() == bootstrap._booster.current_iteration() + 40
    np.testing.assert_array_equal(bootstrap.predict_proba(X[:50]), before)
    assert not np.allclose(updated.predict_proba(X[:50]), before)


def test_continued_training_cannot_learn_a_class_it_never_saw(data, config, bootstrap):
    # Documented limitation (see continue_training): the unseen class's
    # hessian is ~0, so with min_sum_hessian_in_leaf no tree can split on it.
    X, y, _ = data
    unseen_rows = X[y == UNSEEN]
    updated = bootstrap.continue_training(X, y, num_boost_round=40)
    assert (np.argmax(updated.predict_proba(unseen_rows), axis=1) == UNSEEN).mean() == 0.0


def test_continued_training_works_on_a_model_loaded_from_broadcast_bytes(data, config, bootstrap):
    X, y, class_names = data
    loaded = BoostingClassifier.from_bytes(
        bootstrap.to_bytes(), config.boosting, len(class_names), 0, 0, config.cascade.confidence_threshold
    )
    updated = loaded.continue_training(X, y, num_boost_round=5)
    assert updated._booster.current_iteration() == bootstrap._booster.current_iteration() + 5


def test_select_alerts_takes_only_flagged_rows_up_to_the_budget():
    errors = np.array([0.1, 0.9, 0.2, 0.8, 0.95, 0.05])
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, 1, 0, 2, 1, 0])

    everything = select_alerts(errors, 0.5, X, y, budget=10, seed=0)
    capped = select_alerts(errors, 0.5, X, y, budget=2, seed=0)

    np.testing.assert_array_equal(everything.y, [1, 2, 1])
    np.testing.assert_array_equal(everything.X_raw, X[[1, 3, 4]])
    assert len(capped) == 2 and set(capped.y) <= {1, 2}


def test_alerts_round_trip_through_fit_metrics():
    alerts = SurfacedAlerts(np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32), np.array([1, 2, 0, 1]))
    decoded = decode_alerts(encode_alerts(alerts), num_features=3)
    np.testing.assert_array_equal(decoded.X_raw, alerts.X_raw)
    np.testing.assert_array_equal(decoded.y, alerts.y)
    assert decode_alerts({}, 3) is None
    with pytest.raises(ValueError):
        decode_alerts(encode_alerts(alerts), num_features=5)


def test_updater_uses_only_surviving_clients_alerts(data, config, bootstrap):
    X, y, class_names = data
    seen = y != UNSEEN
    evaluated = []
    updater = BoostingUpdater(
        config.boosting, X[seen], y[seen], class_names, evaluate=lambda m: evaluated.append(m) or {"accuracy": 1.0}
    )
    unseen_rows = np.where(y == UNSEEN)[0]
    honest = SurfacedAlerts(X[unseen_rows[:100]], y[unseen_rows[:100]])
    excluded = SurfacedAlerts(X[unseen_rows[100:130]], np.zeros(30, dtype=np.int64))  # poisoned labels

    assert [updater.is_update_round(r) for r in (1, 2, 3, 4)] == [False, True, False, True]
    unchanged, none = updater.update(bootstrap, 2, {0: None, 1: excluded}, surviving_clients={0})
    assert unchanged is bootstrap and none is None and updater.version == 1

    updated, record = updater.update(bootstrap, 2, {0: honest, 1: excluded}, surviving_clients={0})

    assert record.version == updater.version == 2
    # A class the model was never trained on can't be learned by init_model
    # continuation, so this update retrains.
    assert record.mode == "retrain" and record.new_classes == [class_names[UNSEEN]]
    assert record.alerts_used == 100 and record.alerts_by_class == {class_names[UNSEEN]: 100}
    assert record.alerts_from_excluded_clients == 30
    assert record.metrics == {"accuracy": 1.0} and evaluated == [updated]
    # Learned from the alerts: most of the class's rows the alerts didn't include.
    held_back = unseen_rows[:130][100:]  # the excluded client's rows, never trained on
    assert (np.argmax(updated.predict_proba(X[held_back]), axis=1) == UNSEEN).mean() > 0.5

    # Alerts accumulate; known classes continue the model with init_model.
    known = np.where(y == 1)[0][:5]
    again, second = updater.update(updated, 4, {0: SurfacedAlerts(X[known], y[known])}, surviving_clients={0})
    assert updater.total_alerts == 105
    assert second.mode == "continue" and second.new_classes == []
    assert again._booster.current_iteration() == updated._booster.current_iteration() + config.boosting.update_num_boost_round


def test_updates_disabled_when_cadence_is_zero(data, config):
    X, y, class_names = data
    off = BoostingUpdater(dataclasses.replace(config.boosting, update_every_n_rounds=0), X, y, class_names)
    assert not any(off.is_update_round(r) for r in range(1, 10))


def _client(config, data, bootstrap, y_train=True):
    X, y, _ = data
    Xn = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    benign = np.where(y == 0)[0]
    val = benign[:200]
    train = np.setdiff1d(np.arange(len(y)), val)
    client = AutoencoderClient(
        7, Xn[train], X[train], Xn[val], config, X.shape[1], y_train=y[train] if y_train else None
    )
    fit_config = {
        "boosting_model_bytes": bootstrap.to_bytes(), "num_classes": 5, "benign_class": 0, "server_round": 2,
    }
    return client, fit_config, Xn[train], X[train], y[train], Xn[val]


def test_client_surfaces_exactly_its_flagged_boosting_passed_rows(config, data, bootstrap):
    client, fit_config, Xn_train, X_train, y_train, Xn_val = _client(config, data, bootstrap)
    global_weights = get_weights(Autoencoder(Xn_train.shape[1], config.autoencoder.hidden_dims,
                                             config.autoencoder.bottleneck_dim))

    _, _, quiet = client.fit(global_weights, fit_config)
    _, _, metrics = client.fit(global_weights, {**fit_config, SURFACE_ALERTS_KEY: True})

    assert ALERT_COUNT_KEY not in quiet
    alerts = decode_alerts(metrics, X_train.shape[1])
    # Recompute what the client should have flagged with the global weights it received.
    reference = Autoencoder(Xn_train.shape[1], config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    set_weights(reference, global_weights)
    passed = bootstrap.passes_to_autoencoder(X_train)
    threshold = compute_anomaly_threshold(reconstruction_error(reference, Xn_val), config.autoencoder.anomaly_percentile)
    flagged = passed.copy()
    flagged[passed] = reconstruction_error(reference, Xn_train[passed]) > threshold
    expected_count = min(int(flagged.sum()), config.boosting.alert_budget_per_client)

    assert metrics[ALERT_COUNT_KEY] == expected_count > 0
    flagged_rows = {row.tobytes(): label for row, label in zip(X_train[flagged], y_train[flagged])}
    for row, label in zip(alerts.X_raw, alerts.y):
        assert flagged_rows[row.tobytes()] == label


def test_client_without_labels_surfaces_nothing(config, data, bootstrap):
    client, fit_config, Xn_train, *_ = _client(config, data, bootstrap, y_train=False)
    weights = get_weights(Autoencoder(Xn_train.shape[1], config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim))
    _, _, metrics = client.fit(weights, {**fit_config, SURFACE_ALERTS_KEY: True})
    assert metrics[ALERT_COUNT_KEY] == 0


def test_strategy_updates_boosting_from_survivors_and_broadcasts_it_next_round(data, config, bootstrap):
    X, y, class_names = data
    seen = y != UNSEEN
    updater = BoostingUpdater(config.boosting, X[seen], y[seen], class_names)
    strategy = TrustFilteredStrategy(
        config, bootstrap.to_bytes(), len(class_names), 0, boosting_updater=updater,
        fraction_fit=1.0, min_fit_clients=4, min_available_clients=4,
    )
    start = [np.zeros((4, 3)), np.zeros(3)]
    strategy._round_start_weights = start
    rng = np.random.default_rng(0)
    unseen_rows = np.where(y == UNSEEN)[0][:40]
    results = []
    for cid in range(4):
        delta = [np.ones((4, 3)) + rng.normal(0, 0.05, (4, 3)), np.ones(3)]
        metrics = {"client_id": cid, **encode_alerts(SurfacedAlerts(X[unseen_rows[cid * 10:(cid + 1) * 10]],
                                                                    y[unseen_rows[cid * 10:(cid + 1) * 10]]))}
        results.append((_FakeClientProxy(str(cid)), FitRes(Status(Code.OK, "ok"), ndarrays_to_parameters(
            [s + d for s, d in zip(start, delta)]), 100, metrics)))

    before = strategy.boosting_model_bytes
    strategy.aggregate_fit(1, results, [])  # not an update round
    assert strategy.boosting_model_bytes == before and strategy.round_history[-1]["boosting_update"] is None

    strategy._round_start_weights = start
    strategy.aggregate_fit(2, results, [])

    entry = strategy.round_history[-1]
    assert entry["boosting_model_version"] == 1  # what clients filtered with this round
    assert entry["boosting_update"]["version"] == strategy.boosting_model_version == 2
    assert entry["boosting_update"]["alerts_used"] == 10 * len(entry["survivors"])
    assert strategy.boosting_model_bytes != before

    class _Manager:
        def sample(self, num_clients, min_num_clients=None):
            return [_FakeClientProxy(str(i)) for i in range(num_clients)]

        def num_available(self):
            return 4

        def wait_for(self, num_clients, timeout):
            return True

    fit_ins = strategy.configure_fit(3, ndarrays_to_parameters(start), _Manager())
    assert fit_ins[0][1].config["boosting_model_bytes"] == strategy.boosting_model_bytes
    assert fit_ins[0][1].config[SURFACE_ALERTS_KEY] is False
    assert strategy.configure_fit(4, ndarrays_to_parameters(start), _Manager())[0][1].config[SURFACE_ALERTS_KEY] is True


def test_simulation_runs_the_same_update(data, config, bootstrap):
    X, y, class_names = data
    seen = y != UNSEEN
    sim_config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, num_clients=4, dirichlet_alpha=1.0),
        autoencoder=dataclasses.replace(config.autoencoder, local_epochs=1),
    )
    # Clients see the class the bootstrap never learned, so their autoencoders flag it.
    client_data = partition_and_normalize_clients(X, y, 0, sim_config.data, seed=4)
    updater = BoostingUpdater(sim_config.boosting, X[seen], y[seen], class_names)

    result = run_simulated_fl_training(
        client_data, bootstrap, len(class_names), 0, sim_config, num_rounds=3, seed=4, boosting_updater=updater,
    )

    records = [r.boosting_update for r in result.rounds]
    assert records[0] is None and records[2] is None  # update cadence is every 2 rounds
    assert records[1] is not None and records[1].version == 2 and records[1].alerts_used > 0
    # The autoencoders flagged the class boosting never learned, the alerts
    # carried it to the server, and boosting now knows it: zero-day -> known.
    assert records[1].mode == "retrain" and records[1].new_classes == [class_names[UNSEEN]]
    unseen_rows = X[y == UNSEEN]
    assert (np.argmax(bootstrap.predict_proba(unseen_rows), axis=1) == UNSEEN).mean() == 0.0
    assert class_names[UNSEEN] in records[1].alerts_by_class
    # From none of its rows to a clear share, learned from the handful of
    # alerts two short rounds produced (17% at this seed).
    assert (np.argmax(result.final_boosting_model.predict_proba(unseen_rows), axis=1) == UNSEEN).mean() > 0.1


def test_simulation_without_an_updater_keeps_the_bootstrap_model(data, config, bootstrap):
    X, y, class_names = data
    sim_config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, num_clients=3, dirichlet_alpha=1.0),
        autoencoder=dataclasses.replace(config.autoencoder, local_epochs=1),
    )
    client_data = partition_and_normalize_clients(X, y, 0, sim_config.data, seed=5)
    result = run_simulated_fl_training(client_data, bootstrap, len(class_names), 0, sim_config, num_rounds=2, seed=5)
    assert result.final_boosting_model is bootstrap
    assert all(r.boosting_update is None for r in result.rounds)
