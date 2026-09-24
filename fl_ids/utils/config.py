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
