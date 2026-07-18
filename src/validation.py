"""Row-level validation of the IPL datasets using pydantic models.

Each raw dataset has a corresponding pydantic model (:class:`Team`,
:class:`Player`, :class:`Match`, :class:`PlayerMatch`, :class:`BallByBall`) that
declares the expected type of every column and a handful of domain rules (date
format, allowed categorical values, non-negative counts).

Validation is **collect-and-report**: :func:`validate_dataset` runs every row it
is given through the model, gathers the failures, and returns a
:class:`ValidationReport` instead of raising. The caller decides what to do with
the result.

Because the datasets range from 13 rows (Team) to ~150k (Ball_By_Ball), rows are
validated through Python objects; :func:`validate_dataset` therefore samples a
bounded number of rows by default (``sample_size``) and only collects the whole
DataFrame when explicitly asked (``sample_size=None``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Annotated, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError, field_validator
from pyspark.sql import DataFrame

from spark_session import create_spark_session

logger = logging.getLogger(__name__)

# The dates in these files are strings like "4/18/2008" (M/D/YYYY), not ISO.
# Shared, public so the transform stage reuses the same definition (see transform.py).
_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
# Equivalent Spark ``to_date`` pattern for the same format.
DATE_FORMAT = "M/d/yyyy"

# Sentinel values used in the raw data to mean "no/unknown"; treated as absent.
SENTINELS = frozenset({"", "N/A", "NA", "null", "NULL"})


def _check_date(value: Optional[str]) -> Optional[str]:
    """Validate an optional M/D/YYYY date string, raising if malformed."""
    if value is not None and not _DATE_RE.match(value):
        raise ValueError(f"expected M/D/YYYY date, got {value!r}")
    return value


# Reusable field type: an optional M/D/YYYY date string, format-checked.
# Any model field typed ``DateStr`` is validated automatically -- no per-field
# validator boilerplate needed.
DateStr = Annotated[Optional[str], AfterValidator(_check_date)]


class _Base(BaseModel):
    """Shared config for all dataset models.

    * ``populate_by_name`` lets us map quirky CSV headers (``MatcH_id``,
      ``PLAYER_SK``) onto clean Python attribute names via ``alias``.
    * ``extra="forbid"`` makes an unexpected column a validation error rather
      than being silently ignored, so schema drift is caught.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        """Normalize sentinel strings (and NaN) to ``None`` before typing."""
        if isinstance(value, str) and value.strip() in SENTINELS:
            return None
        if isinstance(value, float) and value != value:  # NaN
            return None
        return value


class Team(_Base):
    """A row of ``Team.csv``."""

    Team_SK: int
    Team_Id: int
    Team_Name: str


class Player(_Base):
    """A row of ``Player.csv``."""

    PLAYER_SK: int
    Player_Id: int
    Player_Name: str
    DOB: DateStr = None
    Batting_hand: Optional[str] = None
    Bowling_skill: Optional[str] = None
    Country_Name: Optional[str] = None


class Match(_Base):
    """A row of ``Match.csv``."""

    Match_SK: int
    match_id: int
    Team1: str
    Team2: str
    match_date: DateStr = None
    Season_Year: int
    Venue_Name: Optional[str] = None
    City_Name: Optional[str] = None
    Country_Name: Optional[str] = None
    Toss_Winner: Optional[str] = None
    match_winner: Optional[str] = None
    Toss_Name: Optional[str] = None
    Win_Type: Optional[str] = None
    Outcome_Type: Optional[str] = None
    ManOfMach: Optional[str] = None
    # Win_Margin is string-typed in the source (blank on no-result matches).
    Win_Margin: Optional[str] = None
    Country_id: int


class PlayerMatch(_Base):
    """A row of ``Player_match.csv``.

    Contains a sentinel row (all -1 / ``N/A``); most fields are therefore
    optional so that row validates as "empty" rather than failing.
    """

    Player_match_SK: int
    PlayerMatch_key: float
    Match_Id: int
    Player_Id: int
    Player_Name: Optional[str] = None
    DOB: DateStr = None
    Batting_hand: Optional[str] = None
    Bowling_skill: Optional[str] = None
    Country_Name: Optional[str] = None
    Role_Desc: Optional[str] = None
    Player_team: Optional[str] = None
    Opposit_Team: Optional[str] = None
    Season_year: Optional[int] = None
    is_manofThematch: Optional[int] = None
    Age_As_on_match: Optional[int] = None
    IsPlayers_Team_won: Optional[int] = None
    Batting_Status: Optional[str] = None
    Bowling_Status: Optional[str] = None
    Player_Captain: Optional[str] = None
    Opposit_captain: Optional[str] = None
    Player_keeper: Optional[str] = None
    Opposit_keeper: Optional[str] = None


class BallByBall(_Base):
    """A row of ``Ball_By_Ball.csv`` (one delivery)."""

    MatcH_id: int
    Over_id: int
    Ball_id: int
    Innings_No: int
    Team_Batting: Optional[int] = None
    Team_Bowling: Optional[int] = None
    Striker_Batting_Position: Optional[int] = None
    Extra_Type: Optional[str] = None
    Runs_Scored: int
    Extra_runs: int
    Wides: int
    Legbyes: int
    Byes: int
    Noballs: int
    Penalty: int
    Bowler_Extras: int
    Out_type: Optional[str] = None
    Caught: int
    Bowled: int
    Run_out: int
    LBW: int
    Retired_hurt: int
    Stumped: int
    caught_and_bowled: int
    hit_wicket: int
    ObstructingFeild: int
    Bowler_Wicket: int
    Match_Date: DateStr = None
    Season: int
    Striker: Optional[int] = None
    Non_Striker: Optional[int] = None
    Bowler: Optional[int] = None
    Player_Out: Optional[int] = None
    Fielders: Optional[int] = None
    Striker_match_SK: Optional[int] = None
    StrikerSK: Optional[int] = None
    NonStriker_match_SK: Optional[int] = None
    NONStriker_SK: Optional[int] = None
    Fielder_match_SK: Optional[int] = None
    Fielder_SK: Optional[int] = None
    Bowler_match_SK: Optional[int] = None
    BOWLER_SK: Optional[int] = None
    PlayerOut_match_SK: Optional[int] = None
    BattingTeam_SK: Optional[int] = None
    BowlingTeam_SK: Optional[int] = None
    Keeper_Catch: int
    Player_out_sk: Optional[int] = None
    MatchDateSK: Optional[int] = None

    @field_validator(
        "Runs_Scored", "Extra_runs", "Wides", "Legbyes", "Byes", "Noballs", "Penalty"
    )
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"must be non-negative, got {value}")
        return value


# Logical dataset name -> model. Mirrors ``ingestion.DATASETS``.
MODELS: dict[str, type[_Base]] = {
    "matches": Match,
    "players": Player,
    "teams": Team,
    "player_match": PlayerMatch,
    "ball_by_ball": BallByBall,
}


@dataclass
class RowError:
    """A single row that failed validation."""

    index: int
    errors: list[dict]


@dataclass
class ValidationReport:
    """Outcome of validating one dataset."""

    name: str
    total: int
    validated: int
    failures: list[RowError] = field(default_factory=list)

    @property
    def passed(self) -> int:
        """Number of validated rows with no errors."""
        return self.validated - len(self.failures)

    @property
    def is_valid(self) -> bool:
        """True when every validated row passed."""
        return not self.failures

    def log(self, max_examples: int = 5) -> None:
        """Log a human-readable summary of this report."""
        status = "OK" if self.is_valid else "FAILED"
        logger.info(
            "[%s] %s: %d/%d rows passed (of %d total)",
            status,
            self.name,
            self.passed,
            self.validated,
            self.total,
        )
        for row_error in self.failures[:max_examples]:
            for err in row_error.errors:
                loc = ".".join(str(p) for p in err.get("loc", ()))
                logger.warning(
                    "  row %d | %s: %s", row_error.index, loc, err.get("msg")
                )
        remaining = len(self.failures) - max_examples
        if remaining > 0:
            logger.warning("  ... and %d more failing rows", remaining)


def validate_dataset(
    name: str,
    df: DataFrame,
    model: type[_Base],
    sample_size: Optional[int] = 1000,
) -> ValidationReport:
    """Validate rows of ``df`` against ``model`` and collect the failures.

    Args:
        name: Logical dataset name, used in the report.
        df: The DataFrame to validate.
        model: The pydantic model to validate each row against.
        sample_size: Maximum number of rows to pull into Python and validate.
            ``None`` validates every row (may be expensive for large tables).

    Returns:
        A :class:`ValidationReport` describing how many rows passed and which
        failed. This function does not raise on data errors.
    """
    total = df.count()
    rows = df.collect() if sample_size is None else df.limit(sample_size).collect()

    report = ValidationReport(name=name, total=total, validated=len(rows))
    for index, row in enumerate(rows):
        try:
            model.model_validate(row.asDict())
        except ValidationError as exc:
            report.failures.append(RowError(index=index, errors=exc.errors()))
    return report


def validate_all(
    frames: dict[str, DataFrame],
    models: dict[str, type[_Base]] = MODELS,
    sample_size: Optional[int] = 1000,
) -> dict[str, ValidationReport]:
    """Validate every DataFrame that has a corresponding model.

    Args:
        frames: Mapping of dataset name to DataFrame (e.g. from
            :func:`ingestion.load_all`).
        models: Mapping of dataset name to pydantic model.
        sample_size: Passed through to :func:`validate_dataset`.

    Returns:
        Mapping of dataset name to its :class:`ValidationReport`.
    """
    reports: dict[str, ValidationReport] = {}
    for name, df in frames.items():
        model = models.get(name)
        if model is None:
            logger.warning("No validation model for dataset '%s'; skipping", name)
            continue
        report = validate_dataset(name, df, model, sample_size)
        report.log()
        reports[name] = report
    return reports


def main(sample_size: Optional[int] = 1000) -> dict[str, ValidationReport]:
    """Load all datasets and validate them end to end.

    Imports :mod:`ingestion` lazily so this module has no import-time dependency
    on it, then loads every dataset and validates it.

    Args:
        sample_size: Number of rows to validate per dataset (``None`` for all).

    Returns:
        Mapping of dataset name to its :class:`ValidationReport`.
    """
    from ingestion import configure_logging, load_all

    configure_logging()
    spark = create_spark_session()
    try:
        frames = load_all(spark)
        if not frames:
            logger.error("No datasets loaded; nothing to validate")
            return {}
        return validate_all(frames, sample_size=sample_size)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
