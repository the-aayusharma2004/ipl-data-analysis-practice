"""Load and explore the raw IPL cricket datasets with PySpark.

This module reads the five raw CSV files that make up the IPL dataset and
prints a standard set of exploratory diagnostics for each one:

* shape (rows, columns)
* column names and their inferred types
* the first few rows
* a missing-value summary

It is intended to be run directly (``python ingestion.py`` from ``src/``) or
imported and driven programmatically via :func:`explore`.

The Spark session is provided by :func:`spark_session.create_spark_session`,
which configures a local ``local[*]`` session with an ``ERROR`` log level.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, isnan, when

from spark_session import create_spark_session

logger = logging.getLogger(__name__)

# Path to the raw data, anchored to this file's location rather than the current
# working directory, so the module works regardless of where it is launched from.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"

# Logical dataset name -> CSV filename in DATA_DIR.
DATASETS: dict[str, str] = {
    "matches": "Match.csv",
    "players": "Player.csv",
    "teams": "Team.csv",
    "player_match": "Player_match.csv",
    "ball_by_ball": "Ball_By_Ball.csv",
}

# Spark type names for which ``isnan`` is valid (floating point only).
_FLOAT_TYPES: frozenset[str] = frozenset({"float", "double"})


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a console logging handler for this module.

    Idempotent: if the root logger already has handlers (for example because
    the caller configured logging, or this is called twice), it does nothing.

    Args:
        level: The logging level to set on the root logger.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def load_csv(spark: SparkSession, path: Path) -> DataFrame:
    """Load a single CSV file into a Spark DataFrame.

    Args:
        spark: An active Spark session.
        path: Path to the CSV file to read.

    Returns:
        A DataFrame with header row used for column names and schema inferred.

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file. This
            is checked before handing the path to Spark so the error is a clean
            Python exception rather than an opaque JVM stack trace.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading %s", path.name)
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )


def load_all(
    spark: SparkSession,
    datasets: dict[str, str] = DATASETS,
    data_dir: Path = DATA_DIR,
) -> dict[str, DataFrame]:
    """Load every configured dataset, skipping any whose file is missing.

    A missing file is logged as a warning and skipped rather than aborting the
    whole run, so one absent CSV does not prevent exploring the others.

    Args:
        spark: An active Spark session.
        datasets: Mapping of logical name to CSV filename.
        data_dir: Directory that contains the CSV files.

    Returns:
        Mapping of logical name to loaded DataFrame, containing only the
        datasets that were found and loaded successfully.
    """
    loaded: dict[str, DataFrame] = {}
    for name, filename in datasets.items():
        try:
            loaded[name] = load_csv(spark, data_dir / filename)
        except FileNotFoundError as exc:
            logger.warning("Skipping dataset '%s': %s", name, exc)
    return loaded


def missing_value_counts(df: DataFrame) -> dict[str, int]:
    """Count null (and NaN, where applicable) values per column.

    Nulls are counted for every column. ``NaN`` is additionally counted for
    floating-point columns only, since Spark's ``isnan`` is undefined for
    non-numeric types.

    Args:
        df: The DataFrame to inspect.

    Returns:
        Mapping of column name to its count of missing values.
    """
    def missing(name: str, dtype: str):
        is_null = col(name).isNull()
        if dtype in _FLOAT_TYPES:
            is_null = is_null | isnan(col(name))
        return count(when(is_null, name)).alias(name)

    exprs = [missing(name, dtype) for name, dtype in df.dtypes]
    row = df.select(*exprs).first()
    # row is None only for an empty DataFrame with no columns; guard defensively.
    return {name: int(row[name]) for name in df.columns} if row is not None else {}


def dataframe_summary(name: str, df: DataFrame, n_rows: int = 5) -> None:
    """Print exploratory diagnostics for a single DataFrame.

    Reports shape, column names and types, the first ``n_rows`` rows, and a
    summary of columns that contain missing values.

    Args:
        name: Logical name of the dataset, used in the log header.
        df: The DataFrame to summarize.
        n_rows: Number of leading rows to display.
    """
    n_rows_total = df.count()

    logger.info("=" * 70)
    logger.info("Dataset: %s", name)
    logger.info("Shape: %d rows x %d columns", n_rows_total, len(df.columns))

    logger.info("Columns and types:")
    for column_name, dtype in df.dtypes:
        logger.info("  - %s: %s", column_name, dtype)

    logger.info("First %d rows:", n_rows)
    df.show(n_rows, truncate=False)

    missing = {c: n for c, n in missing_value_counts(df).items() if n > 0}
    if not missing:
        logger.info("Missing values: none")
        return
    logger.info("Missing values (columns with at least one):")
    for column_name, n_missing in missing.items():
        pct = (n_missing / n_rows_total * 100) if n_rows_total else 0.0
        logger.info("  - %s: %d (%.1f%%)", column_name, n_missing, pct)


def explore(datasets: dict[str, str] = DATASETS) -> None:
    """Load and summarize every configured dataset end to end.

    Configures logging, creates a Spark session, loads all datasets, prints a
    summary for each, and always stops the Spark session on exit.

    Args:
        datasets: Mapping of logical name to CSV filename to explore.
    """
    configure_logging()
    spark = create_spark_session()
    try:
        frames = load_all(spark, datasets)
        if not frames:
            logger.error("No datasets could be loaded from %s", DATA_DIR)
            return
        for name, df in frames.items():
            dataframe_summary(name, df)
    finally:
        spark.stop()


if __name__ == "__main__":
    explore()
