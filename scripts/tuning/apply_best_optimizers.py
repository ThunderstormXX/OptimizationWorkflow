from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ORDER = [
    "AdamW",
    "Muon",
    "Scion",
    "SFW-L2",
    "SFW-Ortho",
    "O-SGDM",
    "SGD",
]


def main() -> None:
    root = Path(os.environ.get("TUNING_ROOT", "workflow/tuning"))
    best_specs: dict[str, dict[str, Any]] = {}

    for method_dir in root.iterdir() if root.exists() else []:
        if not method_dir.is_dir():
            continue
        best_path = method_dir / "best.json"
        if not best_path.exists():
            continue
        data = json.loads(best_path.read_text())
        spec = data.get("spec")
        if not spec:
            continue
        best_specs[method_dir.name] = spec

    if not best_specs:
        raise SystemExit("No best.json files found under workflow/tuning")

    ordered: list[dict[str, Any]] = []
    for method in ORDER:
        if method in best_specs:
            ordered.append(best_specs[method])
    # Append any extra methods not in ORDER
    for method, spec in best_specs.items():
        if method not in ORDER:
            ordered.append(spec)

    out_path = Path("configs/tasks/mnist_cnn/optimizers.json")
    out_path.write_text(json.dumps(ordered, indent=2))
    print(f"Wrote {len(ordered)} optimizers to {out_path}")


if __name__ == "__main__":
    main()
