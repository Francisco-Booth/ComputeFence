import os
from pathlib import Path


def check_dataset(dataset_path=None, input_column=None, label_column=None):
    results = []

    if dataset_path is None:
        results.append({
            "status": "warn",
            "message": "No dataset path provided — skipping dataset checks",
            "fix": "Run with --dataset path/to/your/dataset.csv to enable dataset checks"
        })
        return results

    # Check file exists
    path = Path(dataset_path)
    if not path.exists():
        results.append({
            "status": "fail",
            "message": f"Dataset file not found: {dataset_path}",
            "fix": "Check the path and try again"
        })
        return results

    # Check file extension
    if path.suffix not in [".csv", ".parquet", ".json"]:
        results.append({
            "status": "warn",
            "message": f"Unrecognised file type: {path.suffix}. Expected .csv, .parquet, or .json",
            "fix": "Convert your dataset to CSV or Parquet format"
        })
        return results

    # Load the dataset
    try:
        import pandas as pd

        if path.suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        elif path.suffix == ".json":
            df = pd.read_json(dataset_path)

    except Exception as e:
        results.append({
            "status": "fail",
            "message": f"Could not load dataset: {e}",
            "fix": "Check the file is not corrupted and is a valid CSV, Parquet, or JSON file"
        })
        return results

    # Row count
    row_count = len(df)
    results.append({
        "status": "pass",
        "message": f"Dataset loaded — {row_count:,} rows, {len(df.columns)} columns"
    })

    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        results.append({
            "status": "warn",
            "message": f"{missing:,} missing values detected across dataset",
            "fix": "Review and handle missing values before training"
        })
    else:
        results.append({
            "status": "pass",
            "message": "No missing values detected"
        })

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        pct = (duplicate_count / row_count) * 100
        results.append({
            "status": "warn",
            "message": f"{duplicate_count:,} duplicate rows detected ({pct:.1f}% of dataset)",
            "fix": "Remove duplicates with df.drop_duplicates() before training"
        })
    else:
        results.append({
            "status": "pass",
            "message": "No duplicate rows detected"
        })

    # Optional classification checks
    if input_column and label_column:
        if input_column not in df.columns:
            results.append({
                "status": "fail",
                "message": f"Input column '{input_column}' not found in dataset",
                "fix": f"Available columns: {', '.join(df.columns.tolist())}"
            })
        elif label_column not in df.columns:
            results.append({
                "status": "fail",
                "message": f"Label column '{label_column}' not found in dataset",
                "fix": f"Available columns: {', '.join(df.columns.tolist())}"
            })
        else:
            # Check for conflicting labels
            conflicts = df.groupby(input_column)[label_column].nunique()
            conflicting = conflicts[conflicts > 1].count()
            if conflicting > 0:
                results.append({
                    "status": "warn",
                    "message": f"{conflicting:,} inputs have conflicting labels",
                    "fix": "Review and resolve label conflicts before training"
                })
            else:
                results.append({
                    "status": "pass",
                    "message": "No conflicting labels detected"
                })

            # Class distribution
            class_counts = df[label_column].value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")
            if imbalance_ratio > 10:
                results.append({
                    "status": "warn",
                    "message": f"Class imbalance detected — ratio {imbalance_ratio:.1f}:1 between largest and smallest class",
                    "fix": "Consider resampling or using class weights during training"
                })
            else:
                results.append({
                    "status": "pass",
                    "message": f"Class distribution looks reasonable — ratio {imbalance_ratio:.1f}:1"
                })

    return results