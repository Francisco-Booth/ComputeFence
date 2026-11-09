import os
from pathlib import Path


def check_dataset(dataset_path=None, input_column=None, label_column=None):
    results = []

    if dataset_path is None:
        results.append({
            "id": "dataset_provided",
            "status": "warn",
            "message": "No dataset path provided — skipping dataset checks",
            "fix": "Run with --dataset path/to/your/dataset.csv to enable dataset checks"
        })
        return results

    path = Path(dataset_path)
    if not path.exists():
        results.append({
            "id": "dataset_exists",
            "status": "fail",
            "message": f"Dataset file not found: {dataset_path}",
            "fix": "Check the path and try again"
        })
        return results

    if path.suffix not in [".csv", ".parquet", ".json"]:
        results.append({
            "id": "dataset_format",
            "status": "warn",
            "message": f"Unrecognised file type: {path.suffix}. Expected .csv, .parquet, or .json",
            "fix": "Convert your dataset to CSV or Parquet format"
        })
        return results

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
            "id": "dataset_load",
            "status": "fail",
            "message": f"Could not load dataset: {e}",
            "fix": "Check the file is not corrupted and is a valid CSV, Parquet, or JSON file"
        })
        return results

    row_count = len(df)
    results.append({
        "id": "dataset_loaded",
        "status": "pass",
        "message": f"Dataset loaded — {row_count:,} rows, {len(df.columns)} columns"
    })

    missing = df.isnull().sum().sum()
    if missing > 0:
        results.append({
            "id": "dataset_missing_values",
            "status": "warn",
            "message": f"{missing:,} missing values detected across dataset",
            "fix": "Review and handle missing values before training"
        })
    else:
        results.append({
            "id": "dataset_missing_values",
            "status": "pass",
            "message": "No missing values detected"
        })

    # BUG-1: convert to str before duplicated() — prevents crash on list columns
    # (chat-format JSON datasets have list-valued columns which are unhashable)
    try:
        duplicate_count = df.astype(str).duplicated().sum()
        if duplicate_count > 0:
            pct = (duplicate_count / row_count) * 100
            results.append({
                "id": "dataset_duplicates",
                "status": "warn",
                "message": f"{duplicate_count:,} duplicate rows detected ({pct:.1f}% of dataset)",
                "fix": "Remove duplicates with df.drop_duplicates() before training"
            })
        else:
            results.append({
                "id": "dataset_duplicates",
                "status": "pass",
                "message": "No duplicate rows detected"
            })
    except Exception as e:
        results.append({
            "id": "dataset_duplicates",
            "status": "warn",
            "message": f"Could not check for duplicates: {e}",
            "fix": "Check your dataset format"
        })

    if input_column and label_column:
        # BUG-2: escape column names before they appear in messages
        from rich.markup import escape
        safe_input = escape(str(input_column))
        safe_label = escape(str(label_column))
        safe_cols = escape(", ".join(str(c) for c in df.columns.tolist()))

        if input_column not in df.columns:
            results.append({
                "id": "dataset_input_column",
                "status": "fail",
                "message": f"Input column '{safe_input}' not found in dataset",
                "fix": f"Available columns: {safe_cols}"
            })
        elif label_column not in df.columns:
            results.append({
                "id": "dataset_label_column",
                "status": "fail",
                "message": f"Label column '{safe_label}' not found in dataset",
                "fix": f"Available columns: {safe_cols}"
            })
        else:
            # BUG-1: convert to str before groupby() — same unhashable list fix
            try:
                str_df = df.astype(str)
                conflicts = str_df.groupby(input_column)[label_column].nunique()
                conflicting = conflicts[conflicts > 1].count()
                if conflicting > 0:
                    results.append({
                        "id": "dataset_label_conflicts",
                        "status": "warn",
                        "message": f"{conflicting:,} inputs have conflicting labels",
                        "fix": "Review and resolve label conflicts before training"
                    })
                else:
                    results.append({
                        "id": "dataset_label_conflicts",
                        "status": "pass",
                        "message": "No conflicting labels detected"
                    })
            except Exception as e:
                results.append({
                    "id": "dataset_label_conflicts",
                    "status": "warn",
                    "message": f"Could not check label conflicts: {e}",
                    "fix": "Check your dataset format"
                })

            class_counts = df[label_column].value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")
            if imbalance_ratio > 10:
                results.append({
                    "id": "dataset_class_balance",
                    "status": "warn",
                    "message": f"Class imbalance detected — ratio {imbalance_ratio:.1f}:1 between largest and smallest class",
                    "fix": "Consider resampling or using class weights during training"
                })
            else:
                results.append({
                    "id": "dataset_class_balance",
                    "status": "pass",
                    "message": f"Class distribution looks reasonable — ratio {imbalance_ratio:.1f}:1"
                })

    return results
