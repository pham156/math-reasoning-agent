from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


LOSS_COLUMNS = ("step", "epoch", "loss", "learning_rate", "grad_norm")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def loss_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {column: _json_value(entry.get(column)) for column in LOSS_COLUMNS}
        for entry in log_history
        if entry.get("loss") is not None
    ]


def save_training_artifacts(
    output_path: str | Path,
    log_history: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, str | None]:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_history = [_json_value(entry) for entry in log_history]
    rows = loss_rows(log_history)

    metrics_path = output_dir / "train_metrics.json"
    history_path = output_dir / "log_history.jsonl"
    csv_path = output_dir / "loss_curve.csv"
    plot_path = output_dir / "loss_curve.png"

    metrics_path.write_text(json.dumps(_json_value(metrics), indent=2) + "\n", encoding="utf-8")
    with history_path.open("w", encoding="utf-8") as handle:
        for entry in normalized_history:
            handle.write(json.dumps(entry) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if rows:
        x_values = [row["step"] if row["step"] is not None else index for index, row in enumerate(rows)]
        y_values = [float(row["loss"]) for row in rows]
        axis.plot(x_values, y_values, linewidth=1.25)
    else:
        axis.text(0.5, 0.5, "No loss values were logged", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.set_title("QLoRA training loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    return {
        "metrics": str(metrics_path),
        "log_history": str(history_path),
        "loss_csv": str(csv_path),
        "loss_plot": str(plot_path),
    }
