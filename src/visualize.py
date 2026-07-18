"""Render the IPL analyses (:mod:`analysis`) as chart images under ``reports``.

Each chart is produced from one analysis function and saved as a PNG. The module
uses a non-interactive matplotlib backend so it runs headless (e.g. as part of
``pipeline.py``) without a display.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

import analysis

logger = logging.getLogger(__name__)

REPORTS_DIR: Path = Path(__file__).resolve().parent.parent / "reports"

sns.set_theme(style="whitegrid")


def save_fig(fig: "plt.Figure", name: str) -> Path:
    """Save ``fig`` to ``reports/<name>.png`` and close it."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart %s", path)
    return path


def _barh(df, x: str, y: str, title: str, xlabel: str) -> "plt.Figure":
    """Horizontal bar chart helper (largest at top)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    data = df.iloc[::-1]  # so the largest ends up on top
    sns.barplot(data=data, x=x, y=y, ax=ax, hue=y, legend=False, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    return fig


def _line(df, x: str, y: str, title: str, ylabel: str) -> "plt.Figure":
    """Line chart helper for per-season trends."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x=x, y=y, marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Season")
    ax.set_ylabel(ylabel)
    return fig


def main() -> list[Path]:
    """Run every analysis and write its chart. Returns the paths written."""
    from ingestion import configure_logging

    configure_logging()
    tables = analysis._load()
    paths: list[Path] = []

    # --- Team performance ---
    paths.append(
        save_fig(
            _barh(analysis.wins_per_team(tables), "wins", "team",
                  "Matches won by team (2008-2017)", "Wins"),
            "team_wins",
        )
    )
    paths.append(
        save_fig(
            _line(analysis.toss_win_correlation(tables), "season", "toss_winner_won_pct",
                  "Toss winner also won the match (%)", "% of matches"),
            "toss_win_correlation",
        )
    )

    # --- Batting ---
    paths.append(
        save_fig(
            _barh(analysis.top_run_scorers(tables=tables), "runs", "Player_Name",
                  "Top run scorers", "Runs"),
            "top_run_scorers",
        )
    )
    paths.append(
        save_fig(
            _barh(analysis.top_strike_rates(tables=tables), "strike_rate", "Player_Name",
                  "Top strike rates (min 500 balls)", "Strike rate"),
            "top_strike_rates",
        )
    )
    paths.append(
        save_fig(
            _barh(analysis.boundary_counts(tables=tables), "boundaries", "Player_Name",
                  "Most boundaries (4s + 6s)", "Boundaries"),
            "boundary_counts",
        )
    )

    # --- Bowling ---
    paths.append(
        save_fig(
            _barh(analysis.top_wicket_takers(tables=tables), "wickets", "Player_Name",
                  "Top wicket takers", "Wickets"),
            "top_wicket_takers",
        )
    )
    paths.append(
        save_fig(
            _barh(analysis.best_economy_rates(tables=tables), "economy", "Player_Name",
                  "Best economy rates (min 500 balls)", "Economy (runs/over)"),
            "best_economy_rates",
        )
    )
    # Dismissal mix as a pie.
    dm = analysis.dismissal_type_mix(tables)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(dm["count"], labels=dm["dismissal_type"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Dismissal type breakdown")
    paths.append(save_fig(fig, "dismissal_type_mix"))

    # --- Season / venue ---
    paths.append(
        save_fig(
            _line(analysis.runs_per_season(tables), "season", "total_runs",
                  "Total runs per season", "Runs"),
            "runs_per_season",
        )
    )
    paths.append(
        save_fig(
            _barh(analysis.top_scoring_venues(tables=tables), "total_runs", "venue",
                  "Top scoring venues", "Runs"),
            "top_scoring_venues",
        )
    )
    paths.append(
        save_fig(
            _line(analysis.powerplay_run_share(tables), "season", "powerplay_pct",
                  "Powerplay run share per season", "% of runs in overs 1-6"),
            "powerplay_run_share",
        )
    )

    logger.info("Wrote %d charts to %s", len(paths), REPORTS_DIR)
    return paths


if __name__ == "__main__":
    main()
