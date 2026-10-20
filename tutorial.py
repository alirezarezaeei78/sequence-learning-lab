"""Execute and save the deliberately constructed teaching example."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np

from lstm_cell import step, worked_parameters


def run():
    values = worked_parameters()
    hn, cn, gates = step(*values)
    return {
        "kind": "authored synthetic illustration; no training or predictive benchmark",
        "gate_order": ["i", "f", "g", "o"],
        "inputs": {name: value.tolist() for name, value in zip(
            ["x", "h_previous", "c_previous", "weight_ih", "weight_hh", "bias_ih", "bias_hh"], values)},
        "gates_and_terms": {name: value.tolist() for name, value in gates.items()},
        "c_next": cn.tolist(), "h_next": hn.tolist(),
        "memory_only": {str(f): [float(f**t) for t in range(11)] for f in [.5, .8, .99]},
        "memory_only_condition": "c0=1; fixed f; zero candidate write; not a trained network",
        "hardware": {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/2026-10-20.json")
    args = parser.parse_args()
    result = run()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
