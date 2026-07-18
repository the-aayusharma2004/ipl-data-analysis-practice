"""Build the IPL analytics star schema (the "gold" layer) from cleaned data.

Takes the cleaned "silver" DataFrames produced by :mod:`transform` and models
them into a dimensional star schema:

* ``dim_team``       - one row per team
* ``dim_player``     - one row per player
* ``dim_match``      - one row per match
* ``dim_date``       - one row per distinct match date
* ``fact_deliveries``- one row per delivery (ball), with measures and foreign
  keys referencing the dimensions above.

Dimension keys reuse the cleaned source business keys (``Team_Id``,
``Player_Id``, ``match_id``) rather than generating new surrogate keys, since the
transform stage already made those keys clean and unique.

Run as a module (``python star_schema.py``) it loads the processed Parquet,
builds the star, and writes it to ``data/processed/gold``.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_session import create_spark_session

logger = logging.getLogger(__name__)

# Silver datasets required to build the star.
_SOURCE_NAMES = ["teams", "players", "matches", "player_match", "ball_by_ball"]

# Measure / descriptive columns kept on the fact table (besides the foreign keys).
_FACT_MEASURES = [
    "Over_id",
    "Ball_id",
    "Innings_No",
    "Runs_Scored",
    "Extra_runs",
    "Wides",
    "Legbyes",
    "Byes",
    "Noballs",
    "Bowler_Extras",
    "Bowler_Wicket",
    "Out_type",
]


def build_dim_team(frames: dict[str, DataFrame]) -> DataFrame:
    """Build ``dim_team`` from the cleaned teams table."""
    return frames["teams"].select("Team_Id", "Team_Name").dropDuplicates(["Team_Id"])


def build_dim_player(frames: dict[str, DataFrame]) -> DataFrame:
    """Build ``dim_player`` from the cleaned players table."""
    return (
        frames["players"]
        .select(
            "Player_Id",
            "Player_Name",
            "DOB",
            "Batting_hand",
            "Bowling_skill",
            "Country_Name",
        )
        .dropDuplicates(["Player_Id"])
    )


def build_dim_match(frames: dict[str, DataFrame]) -> DataFrame:
    """Build ``dim_match`` from the cleaned matches table."""
    return (
        frames["matches"]
        .select(
            "match_id",
            "Team1",
            "Team2",
            "match_date",
            "Season_Year",
            "Venue_Name",
            "City_Name",
            "Toss_Winner",
            "match_winner",
            "Toss_Name",
            "Win_Type",
            "Win_Margin",
            "Outcome_Type",
            "ManOfMach",
        )
        .dropDuplicates(["match_id"])
    )


def _date_key(date_col: "F.Column") -> "F.Column":
    """Return a yyyymmdd integer key for a date column."""
    return F.date_format(date_col, "yyyyMMdd").cast("int")


def build_dim_date(frames: dict[str, DataFrame]) -> DataFrame:
    """Build ``dim_date`` from the distinct match dates.

    The key (``date_key``) is the date formatted as a ``yyyymmdd`` integer, the
    same convention as the source ``MatchDateSK`` column.
    """
    return (
        frames["matches"]
        .select(F.col("match_date").alias("full_date"))
        .where(F.col("full_date").isNotNull())
        .dropDuplicates(["full_date"])
        .withColumn("date_key", _date_key(F.col("full_date")))
        .withColumn("year", F.year("full_date"))
        .withColumn("month", F.month("full_date"))
        .withColumn("day", F.dayofmonth("full_date"))
        .withColumn("day_of_week", F.date_format("full_date", "EEEE"))
        .select("date_key", "full_date", "year", "month", "day", "day_of_week")
    )


def build_fact_deliveries(frames: dict[str, DataFrame]) -> DataFrame:
    """Build ``fact_deliveries`` (one row per ball) from ball_by_ball.

    Renames the quirky source keys to clean foreign-key names and derives a
    ``date_key`` matching ``dim_date``. Measures are carried through unchanged.
    """
    bbb = frames["ball_by_ball"]
    return bbb.select(
        F.col("MatcH_id").alias("match_id"),
        F.col("Team_Batting").alias("batting_team_id"),
        F.col("Team_Bowling").alias("bowling_team_id"),
        F.col("Striker").alias("striker_id"),
        F.col("Non_Striker").alias("non_striker_id"),
        F.col("Bowler").alias("bowler_id"),
        F.col("Player_Out").alias("player_out_id"),
        F.col("Season").alias("season"),
        _date_key(F.col("Match_Date")).alias("date_key"),
        *_FACT_MEASURES,
    )


def build_star(frames: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Build every dimension and the fact table from the cleaned frames.

    Args:
        frames: Mapping of cleaned dataset name to Spark DataFrame (from
            :func:`transform.transform_all` or the processed Parquet).

    Returns:
        Mapping of star-schema table name to Spark DataFrame.
    """
    return {
        "dim_team": build_dim_team(frames),
        "dim_player": build_dim_player(frames),
        "dim_match": build_dim_match(frames),
        "dim_date": build_dim_date(frames),
        "fact_deliveries": build_fact_deliveries(frames),
    }


def main() -> dict:
    """Load processed Parquet, build the star schema, and write it to gold.

    Returns:
        Mapping of star-schema table name to the Parquet path written.
    """
    from ingestion import configure_logging
    from load import GOLD_DIR, PROCESSED_DIR, read_all_spark, write_all

    configure_logging()
    spark = create_spark_session()
    try:
        frames = read_all_spark(spark, _SOURCE_NAMES, PROCESSED_DIR)
        star = build_star(frames)
        return write_all(star, GOLD_DIR)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
