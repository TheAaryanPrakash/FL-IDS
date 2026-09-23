"""Phase 0 smoke test: confirms pytest is wired up and the package imports."""

import fl_ids


def test_package_importable():
    assert fl_ids is not None


def test_config_loads():
    from fl_ids.utils.config import load_config

    config = load_config()
    assert config.seed == 42
    assert config.data.num_clients > 0
    assert 0.0 < config.data.dirichlet_alpha <= 1.0
    assert config.boosting.confidence_threshold == config.cascade.confidence_threshold
