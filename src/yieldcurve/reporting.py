"""Small helpers for persisting analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_metrics(path: str | Path, update: dict[str, Any]) -> dict[str, Any]:
    """Merge a metrics dict into a JSON file, creating the file if needed.

    Existing keys not present in ``update`` are preserved, so multiple scripts
    can contribute to one ``metrics.json``.

    Args:
        path: Path to the JSON metrics file.
        update: Keys to write, overwriting any existing values.

    Returns:
        The merged metrics dict that was written.
    """
    path = Path(path)
    metrics: dict[str, Any] = {}
    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics.update(update)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
