"""Typed configuration loading for FL-IDS.

All tunable parameters (thresholds, learning rates, round counts, trim
fractions, etc.) are defined in a single YAML file (configs/config.yaml)
and loaded into these dataclasses, rather than being scattered as literals
through the codebase. Every other module should import its parameters from
a `Config` instance instead of hard-coding values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dnn_csv_path: str
    pcap_dir: str
    num_clients: int
    dirichlet_alpha: float
    val_benign_fraction: float
    test_fraction: float
    normalize_per_client: bool
    # {attack_type: re-extracted CSV} for captures whose rows in the dataset
    # are column-shifted (fl_ids.data.repair). Empty loads the CSV as is.
    capture_repairs: dict[str, str] = field(default_factory=dict)
    # Per-client z-scores are clipped to [-normalized_clip, normalized_clip]
    # (fl_ids.data.pipeline.normalize_with_scaler). None disables clipping.
    normalized_clip: float | None = None


@dataclass
class BoostingConfig:
    # Note: no confidence_threshold field here -- the cascade decision
    # rule's confidence threshold has exactly one home, CascadeConfig
    # .confidence_threshold (see fl_ids.models.boosting's module
    # docstring for why a second, same-shaped field here was a bug).
    label_source: str
    calibration_fraction: float
    num_boost_round: int
    learning_rate: float
    num_leaves: int
    broadcast_every_n_rounds: int
    update_every_n_rounds: int
    # Leaf regularization. Without it, multiclass training diverges on real
    # Edge-IIoTset data: near-pure leaves on tiny classes get huge outputs,
    # held-out logloss bottoms out around iteration 16 and then climbs to
    # 7-14, and results swing wildly with which rows are in the training set
    # (see fl_ids.models.boosting's train()). Defaults are the validated values.
    min_sum_hessian_in_leaf: float = 1.0
    lambda_l2: float = 1.0
    # Incremental update (fl_ids.models.boosting_update): every
    # update_every_n_rounds rounds (0 = never), surviving clients surface up
    # to alert_budget_per_client autoencoder-flagged rows with their
    # analyst-confirmed labels, and the server continues training with
    # update_num_boost_round more trees (LightGBM init_model).
    update_num_boost_round: int = 50
    alert_budget_per_client: int = 50
    # Share of the server's calibration set held back to score each
    # boosting version (Phase A), so versions are compared on the same data.
    calibration_eval_fraction: float = 0.2


@dataclass
class AutoencoderConfig:
    bottleneck_dim: int
    hidden_dims: list[int]
    learning_rate: float
    local_epochs: int
    batch_size: int
    anomaly_percentile: float
    reconstruction_error_bins: int
    reconstruction_error_range: tuple[float, float]


@dataclass
class CascadeConfig:
    confidence_threshold: float
    anomaly_confidence_clip: tuple[float, float]


@dataclass
class FLConfig:
    num_rounds: int
    clients_per_round: int
    local_epochs: int
    strategy: str


@dataclass
class RobustnessConfig:
    trim_fraction: float
    norm_clip_multiplier: float
    mad_outlier_threshold: float
    trust_ema_alpha: float
    # What sign-flip test attackers report as their example count:
    # "honest" (their true count) or "max_client" (the largest client's
    # count -- inflating their weight under plain FedAvg, as attackers in
    # the literature do). See fl_ids.robustness.attackers.
    attacker_example_count: str = "max_client"


@dataclass
class SDNConfig:
    block_confidence_threshold: float
    rate_limit_confidence_threshold: float
    controller_host: str
    controller_port: int
    bridge_api_host: str
    bridge_api_port: int


@dataclass
class EvaluationConfig:
    poisoning_fractions: list[float]
    output_dir: str
    # Zero-day experiment (fl_ids.eval.zero_day): attack classes to hold
    # out of all training, one run each. Empty = every non-benign class.
    zero_day_holdout_classes: list[str] = field(default_factory=list)
    # Cap on held-out-class rows scored per run (runtime bound only).
    zero_day_max_holdout_rows: int = 5000
    # rounds_to_convergence: a run has converged once its validation loss
    # stays within this fraction of its total improvement above its best
    # (and ends there).
    convergence_tolerance: float = 0.1


@dataclass
class OrchestrationConfig:
    """Phase A / Phase B entrypoints (component 11, fl_ids.orchestration).

    Defaults let hand-built Configs (tests) omit this section.
    """

    # Where Phase A writes the model bundle Phase B loads.
    artifact_dir: str = "saved_models/phase_a"
    # Phase A working files: per-client data handed to client processes,
    # server history, per-process logs.
    run_dir: str = "runs/phase_a"
    # The Flower server's address. Not 8080: that's the SDN bridge's port.
    server_address: str = "127.0.0.1:9091"
    # The per-round state file the dashboard's training view polls.
    live_state_path: str = "/tmp/fl_ids_training_state.json"
    # Caps the federated pool so CPU-only training finishes in minutes
    # (same cap as the Phase 6 evaluation setup).
    pool_subsample_size: int = 60_000
    # Fraction of clients run as sign-flip attackers in Phase A; 0 for a
    # normal training run, >0 to demo the trust filter live.
    malicious_fraction: float = 0.0
    # How long Phase A waits for the whole Flower run before giving up.
    fl_timeout_seconds: float = 3600.0


@dataclass
class LoggingConfig:
    level: str
    log_dir: str
    log_file: str


@dataclass
class Config:
    seed: int
    data: DataConfig
    boosting: BoostingConfig
    autoencoder: AutoencoderConfig
    cascade: CascadeConfig
    fl: FLConfig
    robustness: RobustnessConfig
    sdn: SDNConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to a config YAML file (see configs/config.yaml).

        Returns:
            A populated, immutable-by-convention Config instance.
        """
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            seed=raw["seed"],
            data=DataConfig(**raw["data"]),
            boosting=BoostingConfig(**raw["boosting"]),
            autoencoder=AutoencoderConfig(
                **{
                    **raw["autoencoder"],
                    "reconstruction_error_range": tuple(
                        raw["autoencoder"]["reconstruction_error_range"]
                    ),
                }
            ),
            cascade=CascadeConfig(
                **{
                    **raw["cascade"],
                    "anomaly_confidence_clip": tuple(
                        raw["cascade"]["anomaly_confidence_clip"]
                    ),
                }
            ),
            fl=FLConfig(**raw["fl"]),
            robustness=RobustnessConfig(**raw["robustness"]),
            sdn=SDNConfig(**raw["sdn"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            logging=LoggingConfig(**raw["logging"]),
            orchestration=OrchestrationConfig(**raw.get("orchestration", {})),
        )


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Load the project config, defaulting to configs/config.yaml.

    Args:
        path: Optional override path to a config YAML file.

    Returns:
        A populated Config instance.
    """
    return Config.from_yaml(path if path is not None else _DEFAULT_CONFIG_PATH)


def write_config_yaml(config: Config, path: str | Path) -> Path:
    """Write a `Config` back out as YAML that `Config.from_yaml` reads identically.

    Phase A's client and server processes load their config from a file,
    so the orchestrator writes the exact in-memory config it's running
    with (including any CLI overrides) instead of pointing them at
    `configs/config.yaml`, which may differ.

    Args:
        config: The config to write.
        path: Output path.

    Returns:
        `path`.
    """
    from dataclasses import asdict

    def _plain(value):
        if isinstance(value, dict):
            return {k: _plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        return value

    path = Path(path)
    path.write_text(yaml.safe_dump(_plain(asdict(config)), sort_keys=False))
    return path
