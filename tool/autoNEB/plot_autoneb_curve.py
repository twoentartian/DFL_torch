import argparse
import csv
import logging
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util


def load_curve_rows(csv_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "sample_index",
            "sample_kind",
            "split",
            "left_pivot_index",
            "right_pivot_index",
            "alpha_local",
            "path_progress",
            "loss",
            "accuracy",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise RuntimeError(f"missing required columns in {csv_path}: {sorted(missing_columns)}")

        for row in reader:
            accuracy_value = row["accuracy"].strip()
            rows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "sample_kind": row["sample_kind"].strip(),
                    "split": row["split"].strip(),
                    "left_pivot_index": int(row["left_pivot_index"]),
                    "right_pivot_index": int(row["right_pivot_index"]),
                    "alpha_local": float(row["alpha_local"]),
                    "path_progress": float(row["path_progress"]),
                    "loss": float(row["loss"]),
                    "accuracy": None if accuracy_value == "" else float(accuracy_value),
                }
            )
    if not rows:
        raise RuntimeError(f"no data rows found in {csv_path}")
    return rows


def group_rows_by_split(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["split"])].append(row)
    for split_name in grouped_rows:
        grouped_rows[split_name].sort(key=lambda item: (float(item["path_progress"]), int(item["sample_index"])))
    return dict(grouped_rows)


def get_pivot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pivot_rows = [row for row in rows if row["sample_kind"] == "pivot"]
    pivot_rows.sort(key=lambda item: (float(item["path_progress"]), int(item["sample_index"])))
    return pivot_rows


def build_default_output_path(csv_path: str) -> str:
    base_name = util.basename_without_extension(os.path.basename(csv_path))
    if base_name.endswith(".csv"):
        base_name = util.basename_without_extension(base_name)
    return os.path.join(os.path.dirname(csv_path), f"{base_name}_plot.png")


def make_title(args, csv_path: str) -> str:
    if args.title is not None:
        return args.title
    return f"AutoNEB Curve: {os.path.basename(csv_path)}"


def plot_curve(rows_by_split, pivot_rows, output_path: str, args) -> None:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axis_loss, axis_accuracy = axes

    split_colors = {
        "train": "tab:blue",
        "test": "tab:orange",
    }

    for split_name, split_rows in sorted(rows_by_split.items()):
        x_values = [float(row["path_progress"]) for row in split_rows]
        loss_values = [float(row["loss"]) for row in split_rows]
        accuracy_values = [row["accuracy"] for row in split_rows]
        color = split_colors.get(split_name, None)

        axis_loss.plot(
            x_values,
            loss_values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{split_name} loss",
            color=color,
        )

        valid_accuracy_points = [
            (x_value, accuracy_value)
            for x_value, accuracy_value in zip(x_values, accuracy_values)
            if accuracy_value is not None
        ]
        if valid_accuracy_points:
            axis_accuracy.plot(
                [item[0] for item in valid_accuracy_points],
                [item[1] for item in valid_accuracy_points],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"{split_name} accuracy",
                color=color,
            )

    pivot_progress = [float(row["path_progress"]) for row in pivot_rows]
    if pivot_progress:
        for pivot_x in pivot_progress:
            axis_loss.axvline(pivot_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.25)
            axis_accuracy.axvline(pivot_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.25)

        first_split_name = sorted(rows_by_split.keys())[0]
        pivot_split_rows = [row for row in rows_by_split[first_split_name] if row["sample_kind"] == "pivot"]
        axis_loss.scatter(
            [float(row["path_progress"]) for row in pivot_split_rows],
            [float(row["loss"]) for row in pivot_split_rows],
            color="black",
            s=18,
            zorder=5,
            label="pivots",
        )
        valid_pivot_accuracy_rows = [row for row in pivot_split_rows if row["accuracy"] is not None]
        if valid_pivot_accuracy_rows:
            axis_accuracy.scatter(
                [float(row["path_progress"]) for row in valid_pivot_accuracy_rows],
                [float(row["accuracy"]) for row in valid_pivot_accuracy_rows],
                color="black",
                s=18,
                zorder=5,
                label="pivots",
            )

    axis_loss.set_ylabel("Loss")
    axis_accuracy.set_ylabel("Accuracy")
    axis_accuracy.set_xlabel("Path Progress")

    axis_loss.set_title(make_title(args, args.csv_path))
    axis_loss.grid(True, alpha=0.3)
    axis_accuracy.grid(True, alpha=0.3)
    axis_loss.legend()
    axis_accuracy.legend()

    if args.accuracy_ylim is not None:
        axis_accuracy.set_ylim(args.accuracy_ylim[0], args.accuracy_ylim[1])

    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show(block=True)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train/test loss and accuracy from an AutoNEB curve CSV.")
    parser.add_argument("csv_path", type=str, help="path to autoneb_curve.csv")
    parser.add_argument("-o", "--output", default=None, help="output image path; defaults to <csv_name>_plot.png")
    parser.add_argument("--title", type=str, default=None, help="optional plot title")
    parser.add_argument("--dpi", type=int, default=150, help="output figure DPI")
    parser.add_argument("--show", action="store_true", help="display the figure after saving it")
    parser.add_argument(
        "--accuracy_ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="optional y-axis limits for the accuracy subplot",
    )

    args = parser.parse_args()
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"{args.csv_path} does not exist")
    if os.path.isdir(args.csv_path):
        raise RuntimeError("expected a CSV file path, not a folder")
    if args.dpi <= 0:
        raise RuntimeError("--dpi must be positive")
    if args.accuracy_ylim is not None and args.accuracy_ylim[0] >= args.accuracy_ylim[1]:
        raise RuntimeError("--accuracy_ylim must satisfy YMIN < YMAX")

    logger = logging.getLogger("plot_autoneb_curve")
    util.set_logging(logger, "autoneb_plot")

    output_path = args.output if args.output is not None else build_default_output_path(args.csv_path)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    curve_rows = load_curve_rows(args.csv_path)
    rows_by_split = group_rows_by_split(curve_rows)
    pivot_rows = get_pivot_rows(curve_rows)
    plot_curve(rows_by_split, pivot_rows, output_path, args)
    logger.info(f"saved AutoNEB curve plot to {output_path}")
