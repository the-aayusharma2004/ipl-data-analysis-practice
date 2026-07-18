"""Persist cleaned IPL DataFrames to Parquet under ``data/processed``.

The transform stage (:mod:`transform`) produces typed, cleaned Spark DataFrames;
this module writes them to columnar Parquet (one file per dataset,
``data/processed/<name>.parquet``) so they can be reloaded quickly with their
schema intact.

Writing goes through pandas/pyarrow (``DataFrame.toPandas().to_parquet(...)``)
rather than Spark's native ``DataFrameWriter``. On Windows, Spark's writer relies
on Hadoop's native ``winutils``/``hadoop.dll``, whose ``NativeIO.access0``
signature does not match the bundled Hadoop client jars, raising
``UnsatisfiedLinkError``. The pyarrow path bypasses Hadoop native IO entirely.
The datasets here are small (<=150k rows) so collecting to the driver via Arrow
is cheap; revisit if the data grows to a scale that no longer fits in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

# Destination for processed ("silver") tables, anchored to this file's location.
PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
# The "gold" analytics layer (star schema) lives under processed/gold.
GOLD_DIR: Path = PROCESSED_DIR / "gold"


def write_parquet(
    df: DataFrame,
    name: str,
    out_dir: Path = PROCESSED_DIR,
) -> Path:
    """Write a single Spark DataFrame to ``out_dir/<name>.parquet``.

    The DataFrame is converted to pandas (via Arrow) and written with the
    pyarrow engine, sidestepping Hadoop native IO on Windows.

    Args:
        df: The Spark DataFrame to persist.
        name: Logical dataset name, used as the output file stem.
        out_dir: Directory the Parquet file is written into (created if absent).

    Returns:
        The path of the Parquet file written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"

    pdf = df.toPandas()
    logger.info("Writing %s (%d rows) to %s", name, len(pdf), path)
    pdf.to_parquet(path, engine="pyarrow", index=False)
    return path


def write_all(
    frames: dict[str, DataFrame],
    out_dir: Path = PROCESSED_DIR,
) -> dict[str, Path]:
    """Write every DataFrame to Parquet, continuing past individual failures.

    A failed write is logged and skipped so one bad dataset does not abort the
    rest of the batch.

    Args:
        frames: Mapping of dataset name to Spark DataFrame.
        out_dir: Destination directory for the Parquet files.

    Returns:
        Mapping of dataset name to the path written, for the datasets that
        succeeded.
    """
    written: dict[str, Path] = {}
    for name, df in frames.items():
        try:
            written[name] = write_parquet(df, name, out_dir)
        except Exception:  # noqa: BLE001 - log and continue with remaining datasets
            logger.exception("Failed to write dataset '%s'", name)
    return written


def read_spark(
    spark: SparkSession,
    name: str,
    in_dir: Path = PROCESSED_DIR,
) -> DataFrame:
    """Read ``in_dir/<name>.parquet`` into a Spark DataFrame.

    Args:
        spark: An active Spark session.
        name: Dataset name (file stem) to read.
        in_dir: Directory the Parquet file lives in.

    Returns:
        The dataset as a Spark DataFrame.
    """
    path = in_dir / f"{name}.parquet"
    logger.info("Reading %s from %s", name, path)
    return spark.read.parquet(str(path))


def read_all_spark(
    spark: SparkSession,
    names: list[str],
    in_dir: Path = PROCESSED_DIR,
) -> dict[str, DataFrame]:
    """Read several Parquet datasets into Spark DataFrames.

    Args:
        spark: An active Spark session.
        names: Dataset names (file stems) to read.
        in_dir: Directory the Parquet files live in.

    Returns:
        Mapping of dataset name to Spark DataFrame.
    """
    return {name: read_spark(spark, name, in_dir) for name in names}


def read_pandas(name: str, in_dir: Path = GOLD_DIR) -> "pd.DataFrame":
    """Read ``in_dir/<name>.parquet`` into a pandas DataFrame.

    Used by the analysis/visualization stages, which work on the small gold
    aggregates and do not need Spark.

    Args:
        name: Dataset name (file stem) to read.
        in_dir: Directory the Parquet file lives in (defaults to the gold layer).

    Returns:
        The dataset as a pandas DataFrame.
    """
    path = in_dir / f"{name}.parquet"
    return pd.read_parquet(path, engine="pyarrow")


def read_all_pandas(names: list[str], in_dir: Path = GOLD_DIR) -> dict[str, "pd.DataFrame"]:
    """Read several Parquet datasets into pandas DataFrames.

    Args:
        names: Dataset names (file stems) to read.
        in_dir: Directory the Parquet files live in (defaults to the gold layer).

    Returns:
        Mapping of dataset name to pandas DataFrame.
    """
    return {name: read_pandas(name, in_dir) for name in names}
