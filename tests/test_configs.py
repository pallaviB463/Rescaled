import os, yaml, tempfile
import run_sim
from dgp import SCENARIOS

CFG = os.path.join(os.path.dirname(__file__), "..", "config")


def test_shipped_configs_have_valid_scenarios():
    for name in ("grid.yaml", "grid_pilot.yaml"):
        cfg = yaml.safe_load(open(os.path.join(CFG, name)))
        with tempfile.TemporaryDirectory() as d:
            tasks = run_sim.grid_tasks(cfg, d)
        assert tasks, name
        assert all(sc in SCENARIOS for sc, _, _ in tasks), name


def test_unquoted_null_in_yaml_is_tolerated():
    cfg = yaml.safe_load("n: 100\ncells:\n  - {scenarios: [null, opposed], zero_rates: [0.9], reps: 1}\n")
    with tempfile.TemporaryDirectory() as d:
        assert {sc for sc, _, _ in run_sim.grid_tasks(cfg, d)} == {"null", "opposed"}
