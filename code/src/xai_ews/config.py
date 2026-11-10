"""Access to the YAML configs in configs/. Code takes parameters from here instead of hardcoding."""

from functools import cache

import yaml

from xai_ews.paths import CONFIGS_DIR


@cache
def data_config() -> dict:
    with open(CONFIGS_DIR / "data.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def feature_config() -> dict:
    """Stage-2 selection rules and the selected feature sets (configs/features.yaml)."""
    with open(CONFIGS_DIR / "features.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
