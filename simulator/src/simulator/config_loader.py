"""YAML scenario loading + calibration merging.

Precedence chain (low to high):

    dataclass defaults  <  calibration artifacts  <  YAML scenario
                         <  env (SIM_*)           <  CLI overrides

- defaults: the provisional values baked into config.py
- calibration: data-derived artifacts (copula.json) produced by the
  ``calibrate`` subcommand - regenerated, never hand-edited
- YAML scenario: the committed per-run experiment knobs
- env/CLI: runtime overrides applied by callers after load

Unknown keys are rejected (typos fail loudly instead of silently
ignoring a knob that was never wired up).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import yaml

from simulator.calibrate import apply_to_config
from simulator.config import SimConfig

# Calibration artifacts live under configs/simulator/calibration/ (repo
# root is three parents above this module: <root>/simulator/src/simulator).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CALIBRATION_DIR = _REPO_ROOT / "configs" / "simulator" / "calibration"
COPULA_ARTIFACT = CALIBRATION_DIR / "copula.json"


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def load_calibration() -> dict[str, Any] | None:
    """Return the fitted copula artifact if present, else None.

    Missing artifacts are fine (fall back to the provisional defaults);
    a corrupt artifact is loud.
    """
    if not COPULA_ARTIFACT.exists():
        return None
    with open(COPULA_ARTIFACT, encoding="utf-8") as f:
        data = json.load(f)
    if "corr_matrix" not in data or "nu" not in data:
        raise ValueError(
            f"{COPULA_ARTIFACT}: missing 'nu'/'corr_matrix' - re-run "
            "'simulator calibrate'")
    return data


def _apply_to_nested(cfg: SimConfig, data: dict[str, Any]) -> None:
    """Recursively fill dataclass fields; reject unknown keys.

    Flat CLI-style keys (n_customers, start_date, ...) are routed through
    SimConfig's flat-kwarg table so YAML can use either spelling.
    """
    for key, value in data.items():
        if isinstance(cfg, SimConfig) and not isinstance(value, dict):
            if key in SimConfig._FLAT_TO_NESTED:
                nested_obj, nested_key = SimConfig._FLAT_TO_NESTED[key]
                setattr(getattr(cfg, nested_obj), nested_key, value)
                continue
        if not hasattr(cfg, key):
            raise ValueError(f"unknown config key: {key}")
        current = getattr(cfg, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply_to_nested(current, value)
        else:
            setattr(cfg, key, value)


def load_config(path: str | pathlib.Path | None) -> SimConfig:
    """Build a SimConfig: defaults -> calibration -> YAML scenario.

    env (SIM_*) and CLI overrides are applied by the caller on top.
    """
    cfg = SimConfig()

    calibration = load_calibration()
    if calibration is not None:
        apply_to_config(cfg, calibration)

    if path is not None:
        _apply_to_nested(cfg, load_yaml(path))

    return cfg