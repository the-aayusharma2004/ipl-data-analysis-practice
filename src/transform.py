"""Clean and conform the raw IPL datasets into typed "silver" tables.

This stage fixes the data-quality issues that :mod:`validation` surfaces on the
raw CSVs and emits trustworthy, correctly typed DataFrames:

* date strings (``M/D/YYYY``) cast to Spark ``DateType``;
* the sentinel row (``Player_Id = -1`` / ``N/A``) dropped from ``player_match``;
* the all-null ``Batting_Status`` / ``Bowling_Status`` columns dropped;
* ``Win_Margin`` cast from string to nullable int;
* ``Team_Batting`` / ``Team_Bowling`` reconciled from a mix of numeric team ids
  and team *names* into a single integer id (recovering the rows that Spark's
  ``inferSchema`` had nulled out);
* ``-1`` sentinel foreign keys in ``ball_by_ball`` normalized to ``NULL``.

Run as a module (``python transform.py``) it executes the full pipeline:
load -> validate (report only) -> transform -> write Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_session import create_spark_session
from validation import DATE_FORMAT

logger = logging.getLogger(__name__)

# Foreign-key / dismissal columns in ball_by_ball that use -1 as "not applicable".
_BBB_SENTINEL_FK_COLS = (
    "Player_Out",
    "Fielders",
    "Striker_match_SK",
    "StrikerSK",
    "NonStriker_match_SK",
    "NONStriker_SK",
    "Fielder_match_SK",
    "Fielder_SK",
    "Bowler_match_SK",
    "BOWLER_SK",
    "PlayerOut_match_SK",
    "Player_out_sk",
)


def to_date_col(name: str) -> "F.Column":
    """Return a column expression parsing ``name`` as an ``M/d/yyyy`` date."""
    return F.to_date(F.col(name), DATE_FORMAT)


def try_int(name: str) -> "F.Column":
    """Cast ``name`` to int, yielding NULL on non-numeric input.

    Uses ``try_cast`` rather than ``cast`` because Spark 4 runs in ANSI mode by
    default, where an invalid ``cast`` (e.g. the literal string ``"NULL"`` or a
    team name) raises instead of producing null.
    """
    return F.expr(f"try_cast(`{name}` as int)")


def _trim_strings(df: DataFrame) -> DataFrame:
    """Trim whitespace on every string column of ``df``."""
    for column, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(column, F.trim(F.col(column)))
    return df


def clean_teams(df: DataFrame) -> DataFrame:
    """Clean ``Team.csv``: trim names and drop duplicate team ids."""
    return _trim_strings(df).dropDuplicates(["Team_Id"])


def clean_players(df: DataFrame) -> DataFrame:
    """Clean ``Player.csv``: trim strings and cast ``DOB`` to a date."""
    return _trim_strings(df).withColumn("DOB", to_date_col("DOB"))


def clean_matches(df: DataFrame) -> DataFrame:
    """Clean ``Match.csv``: cast ``match_date`` and ``Win_Margin``, trim strings."""
    return (
        _trim_strings(df)
        .withColumn("match_date", to_date_col("match_date"))
        .withColumn("Win_Margin", try_int("Win_Margin"))
    )


def clean_player_match(df: DataFrame) -> DataFrame:
    """Clean ``Player_match.csv``.

    Drops the sentinel row (``Player_Id = -1``) and the two all-null status
    columns, then casts ``DOB`` to a date.
    """
    return (
        _trim_strings(df)
        .filter(F.col("Player_Id") != -1)
        .drop("Batting_Status", "Bowling_Status")
        .withColumn("DOB", to_date_col("DOB"))
    )


def clean_ball_by_ball(df: DataFrame, teams_df: DataFrame) -> DataFrame:
    """Clean ``Ball_By_Ball.csv``.

    ``Team_Batting`` / ``Team_Bowling`` arrive as strings mixing numeric team ids
    with team *names*. Each is resolved to an integer id: values that are already
    numeric are kept; values that match a team name are mapped to that team's id
    via ``teams_df``. ``Match_Date`` is cast to a date and ``-1`` foreign-key
    sentinels are normalized to ``NULL``.

    Args:
        df: The raw ball-by-ball DataFrame.
        teams_df: The (cleaned) teams DataFrame, used as a name -> id lookup.

    Returns:
        The cleaned ball-by-ball DataFrame.
    """
    df = _trim_strings(df)

    # Build name -> id lookup from the teams dimension.
    lookup = teams_df.select(
        F.col("Team_Name").alias("_team_name"),
        F.col("Team_Id").alias("_team_id"),
    )

    for col_name in ("Team_Batting", "Team_Bowling"):
        # Numeric strings cast cleanly; names try_cast to NULL, then get filled
        # from the lookup join. coalesce keeps whichever resolved.
        numeric_id = try_int(col_name)
        df = (
            df.join(
                lookup,
                on=F.col(col_name) == F.col("_team_name"),
                how="left",
            )
            .withColumn(col_name, F.coalesce(numeric_id, F.col("_team_id")))
            .drop("_team_name", "_team_id")
        )

    df = df.withColumn("Match_Date", to_date_col("Match_Date"))

    for col_name in _BBB_SENTINEL_FK_COLS:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name) == -1, None).otherwise(F.col(col_name)),
            )
    return df


def transform_all(frames: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Clean every dataset present in ``frames``.

    ``ball_by_ball`` requires the cleaned ``teams`` frame for its name -> id
    reconciliation, so teams is cleaned first. Datasets without a cleaner are
    logged and skipped.

    Args:
        frames: Mapping of dataset name to raw DataFrame (from
            :func:`ingestion.load_all`).

    Returns:
        Mapping of dataset name to cleaned DataFrame.
    """
    cleaned: dict[str, DataFrame] = {}

    # Teams first: ball_by_ball's cleaner depends on it.
    if "teams" in frames:
        cleaned["teams"] = clean_teams(frames["teams"])

    single_arg_cleaners = {
        "players": clean_players,
        "matches": clean_matches,
        "player_match": clean_player_match,
    }
    for name, cleaner in single_arg_cleaners.items():
        if name in frames:
            cleaned[name] = cleaner(frames[name])

    if "ball_by_ball" in frames:
        teams_df = cleaned.get("teams", frames.get("teams"))
        if teams_df is None:
            logger.warning(
                "ball_by_ball present but no teams frame; team ids will not be "
                "reconciled from names"
            )
        else:
            cleaned["ball_by_ball"] = clean_ball_by_ball(
                frames["ball_by_ball"], teams_df
            )

    for name in frames:
        if name not in cleaned:
            logger.warning("No cleaner for dataset '%s'; skipping", name)

    return cleaned


def main() -> dict[str, Path]:
    """Run the full pipeline: load -> validate (report) -> transform -> write.

    Validation is run and its report logged, but it is **non-blocking**: the
    known raw-data issues are exactly what this stage fixes, so a failing report
    must not prevent the clean. Imports of the other stages are lazy to avoid
    import-time coupling.

    Returns:
        Mapping of dataset name to the Parquet path written.
    """
    from ingestion import configure_logging, load_all
    from load import write_all
    from validation import validate_all

    configure_logging()
    spark = create_spark_session()
    try:
        frames = load_all(spark)
        if not frames:
            logger.error("No datasets loaded; nothing to transform")
            return {}

        logger.info("Running validation (report only) before transform")
        validate_all(frames)

        cleaned = transform_all(frames)
        return write_all(cleaned)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
