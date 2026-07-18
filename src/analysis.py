"""Analytical queries over the IPL star schema (gold layer).

Each function loads the relevant gold Parquet tables (via :mod:`load`) and
returns a small, tidy pandas DataFrame — one aggregate result per question. No
plotting happens here; :mod:`visualize` turns these results into charts and the
notebook renders them inline. Keeping the queries pure and chart-free means the
same functions back both outputs.

Functions are grouped into four themes: team performance, batting, bowling, and
season/venue trends.
"""

from __future__ import annotations

import pandas as pd

from load import read_pandas

# Cricket: a boundary four scores 4 runs off the bat, a six scores 6.
_FOUR = 4
_SIX = 6
# Powerplay is the first 6 overs of an innings (over ids 1..6 in this data).
_POWERPLAY_OVERS = 6


def _load() -> dict[str, pd.DataFrame]:
    """Load the star-schema tables needed across the analyses."""
    names = ["dim_team", "dim_player", "dim_match", "dim_date", "fact_deliveries"]
    return {name: read_pandas(name) for name in names}


# --------------------------------------------------------------------------- #
# Team performance
# --------------------------------------------------------------------------- #
def wins_per_team(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Total matches won by each team, most wins first."""
    tables = tables or _load()
    matches = tables["dim_match"]
    wins = (
        matches["match_winner"]
        .dropna()
        .value_counts()
        .rename_axis("team")
        .reset_index(name="wins")
    )
    return wins


def wins_per_season(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Matches won by each team per season (long/tidy form)."""
    tables = tables or _load()
    matches = tables["dim_match"]
    out = (
        matches.dropna(subset=["match_winner"])
        .groupby(["Season_Year", "match_winner"])
        .size()
        .reset_index(name="wins")
        .rename(columns={"match_winner": "team", "Season_Year": "season"})
    )
    return out.sort_values(["season", "wins"], ascending=[True, False])


def toss_win_correlation(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Rate at which the toss winner also won the match, per season."""
    tables = tables or _load()
    m = tables["dim_match"].dropna(subset=["Toss_Winner", "match_winner"]).copy()
    m["toss_won_match"] = m["Toss_Winner"] == m["match_winner"]
    out = (
        m.groupby("Season_Year")["toss_won_match"]
        .mean()
        .mul(100)
        .reset_index(name="toss_winner_won_pct")
        .rename(columns={"Season_Year": "season"})
    )
    return out


# --------------------------------------------------------------------------- #
# Batting
# --------------------------------------------------------------------------- #
def top_run_scorers(
    n: int = 10, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Top ``n`` batters by total runs scored off the bat."""
    tables = tables or _load()
    fact, players = tables["fact_deliveries"], tables["dim_player"]
    runs = (
        fact.groupby("striker_id")["Runs_Scored"]
        .sum()
        .reset_index(name="runs")
        .merge(players[["Player_Id", "Player_Name"]], left_on="striker_id", right_on="Player_Id")
    )
    return runs.nlargest(n, "runs")[["Player_Name", "runs"]].reset_index(drop=True)


def top_strike_rates(
    n: int = 10, min_balls: int = 500, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Top ``n`` batters by strike rate (runs per 100 balls faced).

    Only batters who faced at least ``min_balls`` legal-ish deliveries are
    ranked, to avoid tiny-sample outliers.
    """
    tables = tables or _load()
    fact, players = tables["fact_deliveries"], tables["dim_player"]
    grouped = fact.groupby("striker_id").agg(
        runs=("Runs_Scored", "sum"), balls=("Ball_id", "count")
    )
    grouped = grouped[grouped["balls"] >= min_balls].copy()
    grouped["strike_rate"] = grouped["runs"] / grouped["balls"] * 100
    out = (
        grouped.reset_index()
        .merge(players[["Player_Id", "Player_Name"]], left_on="striker_id", right_on="Player_Id")
        .nlargest(n, "strike_rate")
    )
    return out[["Player_Name", "strike_rate", "runs", "balls"]].reset_index(drop=True)


def boundary_counts(
    n: int = 10, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Top ``n`` batters by number of fours and sixes hit."""
    tables = tables or _load()
    fact, players = tables["fact_deliveries"], tables["dim_player"]
    fact = fact.copy()
    fact["fours"] = (fact["Runs_Scored"] == _FOUR).astype(int)
    fact["sixes"] = (fact["Runs_Scored"] == _SIX).astype(int)
    grouped = (
        fact.groupby("striker_id")[["fours", "sixes"]]
        .sum()
        .reset_index()
        .merge(players[["Player_Id", "Player_Name"]], left_on="striker_id", right_on="Player_Id")
    )
    grouped["boundaries"] = grouped["fours"] + grouped["sixes"]
    return grouped.nlargest(n, "boundaries")[
        ["Player_Name", "fours", "sixes", "boundaries"]
    ].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Bowling
# --------------------------------------------------------------------------- #
def top_wicket_takers(
    n: int = 10, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Top ``n`` bowlers by wickets credited (``Bowler_Wicket`` flag)."""
    tables = tables or _load()
    fact, players = tables["fact_deliveries"], tables["dim_player"]
    wk = (
        fact.groupby("bowler_id")["Bowler_Wicket"]
        .sum()
        .reset_index(name="wickets")
        .merge(players[["Player_Id", "Player_Name"]], left_on="bowler_id", right_on="Player_Id")
    )
    return wk.nlargest(n, "wickets")[["Player_Name", "wickets"]].reset_index(drop=True)


def best_economy_rates(
    n: int = 10, min_balls: int = 500, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Best ``n`` bowlers by economy rate (runs conceded per over).

    Runs conceded = runs off the bat plus wides and no-balls (byes/leg-byes are
    not charged to the bowler). Only bowlers with at least ``min_balls`` balls.
    """
    tables = tables or _load()
    fact, players = tables["fact_deliveries"], tables["dim_player"]
    fact = fact.copy()
    fact["conceded"] = fact["Runs_Scored"] + fact["Wides"] + fact["Noballs"]
    grouped = fact.groupby("bowler_id").agg(
        conceded=("conceded", "sum"), balls=("Ball_id", "count")
    )
    grouped = grouped[grouped["balls"] >= min_balls].copy()
    grouped["economy"] = grouped["conceded"] / (grouped["balls"] / 6)
    out = (
        grouped.reset_index()
        .merge(players[["Player_Id", "Player_Name"]], left_on="bowler_id", right_on="Player_Id")
        .nsmallest(n, "economy")
    )
    return out[["Player_Name", "economy", "conceded", "balls"]].reset_index(drop=True)


def dismissal_type_mix(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Count of dismissals by type (excluding the 'no wicket' marker)."""
    tables = tables or _load()
    fact = tables["fact_deliveries"]
    out = (
        fact.loc[fact["Out_type"].str.lower() != "not applicable", "Out_type"]
        .value_counts()
        .rename_axis("dismissal_type")
        .reset_index(name="count")
    )
    return out


# --------------------------------------------------------------------------- #
# Season / venue trends
# --------------------------------------------------------------------------- #
def runs_per_season(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Total runs scored (off the bat) per season."""
    tables = tables or _load()
    fact = tables["fact_deliveries"]
    return (
        fact.groupby("season")["Runs_Scored"]
        .sum()
        .reset_index(name="total_runs")
        .sort_values("season")
    )


def top_scoring_venues(
    n: int = 10, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Top ``n`` venues by total runs scored there (via dim_match)."""
    tables = tables or _load()
    fact, matches = tables["fact_deliveries"], tables["dim_match"]
    joined = fact.merge(
        matches[["match_id", "Venue_Name"]], on="match_id", how="left"
    )
    out = (
        joined.groupby("Venue_Name")["Runs_Scored"]
        .sum()
        .reset_index(name="total_runs")
        .rename(columns={"Venue_Name": "venue"})
        .nlargest(n, "total_runs")
        .reset_index(drop=True)
    )
    return out


def powerplay_run_share(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Share of total runs scored in the powerplay (overs 1-6), per season."""
    tables = tables or _load()
    fact = tables["fact_deliveries"].copy()
    fact["is_powerplay"] = fact["Over_id"] <= _POWERPLAY_OVERS
    grouped = fact.groupby("season").apply(
        lambda g: pd.Series(
            {
                "powerplay_runs": g.loc[g["is_powerplay"], "Runs_Scored"].sum(),
                "total_runs": g["Runs_Scored"].sum(),
            }
        ),
        include_groups=False,
    )
    grouped["powerplay_pct"] = (
        grouped["powerplay_runs"] / grouped["total_runs"] * 100
    )
    return grouped.reset_index()[["season", "powerplay_pct"]]
