"""YAML scenario loading + calibration merging (Phase 1)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from simulator import config_loader
from simulator.config_loader import load_config

_IDENTITY7 = [[1.0 if i == j else 0.0 for j in range(7)] for i in range(7)]


def test_load_config_yaml_overrides_defaults(tmp_path):
    y = tmp_path / "scenario.yaml"
    y.write_text(
        "seed: 7\n"
        "entities:\n"
        "  n_customers: 1234\n"
        "  target_default_rate: 0.05\n"
        "time:\n"
        "  start_date: 2021-05-01\n"
        "problems:\n"
        "  enabled: false\n"
    )
    cfg = load_config(y)
    assert cfg.seed == 7
    assert cfg.entities.n_customers == 1234
    assert cfg.entities.target_default_rate == 0.05
    assert cfg.time.start_date == date(2021, 5, 1)
    assert cfg.problems.enabled is False


def test_load_config_flat_keys_route_to_nested(tmp_path):
    y = tmp_path / "scenario.yaml"
    y.write_text("n_customers: 500\nstart_date: 2022-03-01\n")
    cfg = load_config(y)
    assert cfg.entities.n_customers == 500
    assert cfg.time.start_date == date(2022, 3, 1)


def test_load_config_rejects_unknown_keys(tmp_path):
    y = tmp_path / "scenario.yaml"
    y.write_text("no_such_knob: 1\n")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(y)


def test_load_config_defaults_without_yaml():
    cfg = load_config(None)
    assert cfg.seed == 42
    assert cfg.entities.n_customers == 50000


def test_calibration_merges_then_yaml_wins(tmp_path, monkeypatch):
    cal = tmp_path / "copula.json"
    cal.write_text(json.dumps(
        {"nu": 2.5, "corr_matrix": _IDENTITY7, "gaussian_preferred": False}))
    monkeypatch.setattr(config_loader, "COPULA_ARTIFACT", cal)

    # no YAML: calibration values land in the config
    cfg = load_config(None)
    assert cfg.copula.nu == 2.5

    # YAML overrides calibration (scenario experiment wins)
    y = tmp_path / "scenario.yaml"
    y.write_text("copula:\n  nu: 30.0\n")
    cfg2 = load_config(y)
    assert cfg2.copula.nu == 30.0


def test_calibration_absent_falls_back_to_defaults(tmp_path, monkeypatch):
    missing = tmp_path / "copula.json"
    monkeypatch.setattr(config_loader, "COPULA_ARTIFACT", missing)
    cfg = load_config(None)
    assert cfg.copula.nu == 30.0