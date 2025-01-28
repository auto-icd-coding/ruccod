import subprocess
import pandas as pd
from pathlib import Path

DATASETS = [
    "mkb-full",
    "mkb-full-ehr-train",
    "nerel-bio",
    "ruccon",
]
RESULTS_DIR = Path("./results")


def evaluate_model(dataset, embedding_name, model_name):
    """
    Evaluate a specific model on a specific dataset.
    """
    ref_data_dir = Path("./datasets/test")
    pred_data_dir = RESULTS_DIR / dataset / embedding_name / model_name
    output_path = pred_data_dir / "metrics.txt"

    ref_ann_files = list(ref_data_dir.glob("*.ann"))
    pred_ann_files = list(pred_data_dir.glob("*.ann"))
    if len(ref_ann_files) != len(pred_ann_files):
        print(
            f"Skipping evaluation for {dataset}/{model_name} due to incorrect number of annotation files: {len(pred_ann_files)}"
        )
        return

    command = [
        "python",
        "evaluate_llm_predictions.py",
        "--ref_data_dir",
        str(ref_data_dir),
        "--pred_data_dir",
        str(pred_data_dir),
        "--output_path",
        str(output_path),
    ]

    subprocess.run(command, check=True)


def aggregate_results(datasets):
    """Aggregate results from all datasets and models into a pandas DataFrame."""
    aggregated_results = []

    for dataset in datasets:
        dataset_dir = RESULTS_DIR / dataset

        for embedding_dir in dataset_dir.iterdir():
            if embedding_dir.is_dir():
                for model_dir in embedding_dir.iterdir():
                    if model_dir.is_dir():
                        metrics_path = model_dir / "metrics.txt"
                        if metrics_path.exists():
                            with open(metrics_path, "r") as f:
                                metrics = {}
                                for line in f:
                                    metric_name, score = line.strip().split("\t")
                                    metrics[metric_name] = round(float(score), 4)

                                row = {
                                    "dataset_embedding_model": f"{dataset}_{embedding_dir.name}_{model_dir.name}",
                                    **metrics,
                                }
                                aggregated_results.append(row)

    df = pd.DataFrame(aggregated_results)
    df.set_index("dataset_embedding_model", inplace=True)
    df.to_csv(RESULTS_DIR / "aggregated_metrics.csv")

    return df


def main():
    for dataset in DATASETS:
        dataset_dir = RESULTS_DIR / dataset
        if dataset_dir.is_dir():
            for embedding_dir in dataset_dir.iterdir():
                if embedding_dir.is_dir():
                    for model_dir in embedding_dir.iterdir():
                        if model_dir.is_dir():
                            evaluate_model(dataset, embedding_dir.name, model_dir.name)

    aggregate_results(DATASETS)


if __name__ == "__main__":
    main()
