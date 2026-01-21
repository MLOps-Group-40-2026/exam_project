"""Data drift detection using Evidently.

This script compares the distribution of incoming API predictions
against reference data from training to detect data drift.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests.data_drift_tests import TestNumberOfDriftedColumns
from loguru import logger


# Feature columns (must match those logged in API)
FEATURE_COLUMNS = ["brightness", "contrast", "r_mean", "g_mean", "b_mean"]
TARGET_COLUMN = "prediction"


def create_reference_data() -> pd.DataFrame:
    """Create reference data from training images.

    In production, this would load features extracted from actual training data.
    For now, we create synthetic reference data based on expected distributions.
    """
    # Typical values for coffee leaf images (RGB, 0-255 scale)
    # TODO:  compute from training data
    import numpy as np

    np.random.seed(42)
    n_samples = 500

    reference_data = pd.DataFrame(
        {
            "brightness": np.random.normal(120, 30, n_samples),  # Medium brightness
            "contrast": np.random.normal(50, 15, n_samples),  # Moderate contrast
            "r_mean": np.random.normal(100, 25, n_samples),  # Red channel
            "g_mean": np.random.normal(130, 25, n_samples),  # Green channel (leaves)
            "b_mean": np.random.normal(90, 20, n_samples),  # Blue channel
            "prediction": np.random.choice(
                ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"],
                n_samples,
                p=[0.4, 0.15, 0.15, 0.15, 0.15],  # assume 40% healthy baseline
            ),
        }
    )

    return reference_data


def load_predictions_from_gcs(
    bucket_name: str = "mlops-group-40-2026-dvc",
    prefix: str = "predictions/",
    max_files: int = 100,
    hours_back: int | None = 24,
) -> pd.DataFrame:
    """Load recent prediction logs from GCS.

    Args:
        bucket_name: GCS bucket name
        prefix: Path prefix for prediction files
        max_files: Maximum number of files to load
        hours_back: Only load files from the last N hours (None for all)

    Returns:
        DataFrame with prediction data
    """
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        # filter by time if specified
        if hours_back:
            cutoff = datetime.utcnow() - timedelta(hours=hours_back)
            blobs = [b for b in blobs if b.time_created.replace(tzinfo=None) > cutoff]

        # newest first and limit
        blobs = sorted(blobs, key=lambda b: b.time_created, reverse=True)[:max_files]

        if not blobs:
            logger.warning(f"No prediction files found in gs://{bucket_name}/{prefix}")
            return pd.DataFrame()

        # load and parse JSON files
        records = []
        for blob in blobs:
            try:
                content = blob.download_as_text()
                data = json.loads(content)
                record = {
                    "timestamp": data.get("timestamp"),
                    "prediction": data.get("prediction"),
                    "confidence": data.get("confidence"),
                    **data.get("features", {}),
                }
                records.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse {blob.name}: {e}")

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} predictions from GCS")
        return df

    except Exception as e:
        logger.error(f"Failed to load predictions from GCS: {e}")
        return pd.DataFrame()


def load_predictions_from_local(path: str = "predictions/") -> pd.DataFrame:
    """Load predictions from local directory (for testing)."""
    records = []
    for file_path in Path(path).rglob("*.json"):
        try:
            with open(file_path) as f:
                data = json.load(f)
                record = {
                    "timestamp": data.get("timestamp"),
                    "prediction": data.get("prediction"),
                    "confidence": data.get("confidence"),
                    **data.get("features", {}),
                }
                records.append(record)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

    return pd.DataFrame(records)


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "drift_report.html",
) -> Report:
    """Generate an Evidently drift report.

    Args:
        reference_data: Training/reference data distribution
        current_data: Recent production data
        output_path: Where to save the HTML report

    Returns:
        Evidently Report object
    """
    # ensure we have the right columns
    feature_cols = [c for c in FEATURE_COLUMNS if c in current_data.columns]

    if not feature_cols:
        logger.error("No feature columns found in current data")
        raise ValueError("No feature columns found")

    report = Report(
        metrics=[
            DataDriftPreset(columns=feature_cols),
            TargetDriftPreset(columns=[TARGET_COLUMN]),
        ]
    )

    report.run(
        reference_data=reference_data[feature_cols + [TARGET_COLUMN]],
        current_data=current_data[feature_cols + [TARGET_COLUMN]],
    )

    report.save_html(output_path)
    logger.info(f"Drift report saved to {output_path}")

    return report


def run_drift_tests(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> TestSuite:
    """Run drift detection tests.

    Args:
        reference_data: Training/reference data distribution
        current_data: Recent production data

    Returns:
        TestSuite with pass/fail results
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in current_data.columns]

    tests = TestSuite(
        tests=[
            TestNumberOfDriftedColumns(lt=3),  # Fail if 3+ columns drifted
        ]
    )

    tests.run(
        reference_data=reference_data[feature_cols + [TARGET_COLUMN]],
        current_data=current_data[feature_cols + [TARGET_COLUMN]],
    )

    return tests


def main():
    """Main function to run drift detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Run data drift detection")
    parser.add_argument("--local", action="store_true", help="Use local predictions instead of GCS")
    parser.add_argument("--output", default="drift_report.html", help="Output path for HTML report")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to analyze")
    parser.add_argument("--max-files", type=int, default=100, help="Max prediction files to load")
    args = parser.parse_args()

    logger.info("Starting drift detection analysis")

    # Load reference data (training distribution)
    logger.info("Loading reference data...")
    reference_data = create_reference_data()
    logger.info(f"Reference data: {len(reference_data)} samples")

    # load current production data
    logger.info("Loading current prediction data...")
    if args.local:
        current_data = load_predictions_from_local()
    else:
        current_data = load_predictions_from_gcs(hours_back=args.hours, max_files=args.max_files)

    if current_data.empty:
        logger.warning("No current data available. Creating synthetic data for demo.")
        # slightly drifted synthetic data for demonstration
        import numpy as np

        np.random.seed(123)
        n_samples = 50
        current_data = pd.DataFrame(
            {
                "brightness": np.random.normal(140, 35, n_samples),  # brighter
                "contrast": np.random.normal(55, 15, n_samples),
                "r_mean": np.random.normal(110, 25, n_samples),
                "g_mean": np.random.normal(125, 25, n_samples),
                "b_mean": np.random.normal(95, 20, n_samples),
                "prediction": np.random.choice(
                    ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"],
                    n_samples,
                    p=[0.3, 0.2, 0.2, 0.15, 0.15],  # less healthy more disease
                ),
            }
        )

    logger.info(f"Current data: {len(current_data)} samples")

    # drift report
    logger.info("Generating drift report...")
    run_drift_report(reference_data, current_data, args.output)

    # run tests
    logger.info("Running drift tests...")
    tests = run_drift_tests(reference_data, current_data)

    # test results
    test_results = tests.as_dict()
    print("\n" + "=" * 50)
    print("DRIFT DETECTION RESULTS")
    print("=" * 50)

    for test in test_results.get("tests", []):
        status = "✅ PASS" if test.get("status") == "SUCCESS" else "❌ FAIL"
        print(f"{status}: {test.get('name')}")

    print(f"\nFull report saved to: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
