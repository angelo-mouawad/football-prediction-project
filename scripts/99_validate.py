import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import INTERIM_DIR, MATCHES_PER_SEASON, XG_FIRST_SEASON
from src.teams import to_canonical

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def warn(name: str, detail: str) -> None:
    print(f"  WARN  {name}  {detail}")
    WARNINGS.append(name)


def validate_matches() -> pd.DataFrame | None:
    path = INTERIM_DIR / "matches_football_data.csv"
    print(f"\n{path.name}")
    if not path.exists():
        check("file exists", False, "run scripts/01_fetch_results.py first")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    check("file exists", True)

    # Season completeness. The current season is allowed to be partial.
    counts = df.groupby("season_start").size()
    latest = counts.index.max()
    for year, n in counts.items():
        if year == latest and n < MATCHES_PER_SEASON:
            warn(f"season {year} count", f"{n} matches (season in progress)")
        else:
            check(f"season {year} has {MATCHES_PER_SEASON} matches",
                  n == MATCHES_PER_SEASON, f"got {n}")

    # Duplicates.
    dupes = df.duplicated(subset=["date", "home_team", "away_team"]).sum()
    check("no duplicate fixtures", dupes == 0, f"{dupes} duplicates")

    # A team never plays itself.
    self_play = (df["home_team"] == df["away_team"]).sum()
    check("no team plays itself", self_play == 0, f"{self_play} rows")

    # Dates sort and are contiguous enough.
    check("dates parsed", df["date"].isna().sum() == 0)
    check("dates sorted", df["date"].is_monotonic_increasing)

    # Core columns have no nulls.
    for col in ["home_goals", "away_goals", "result", "home_team", "away_team"]:
        nulls = df[col].isna().sum()
        check(f"{col} has no nulls", nulls == 0, f"{nulls} nulls")

    # Result label agrees with the goals. This catches source corruption.
    derived = df.apply(
        lambda r: "H" if r.home_goals > r.away_goals
        else ("A" if r.home_goals < r.away_goals else "D"),
        axis=1,
    )
    mismatch = (derived != df["result"]).sum()
    check("result matches goals", mismatch == 0, f"{mismatch} mismatches")

    # Every club resolves.
    bad = {t for t in set(df["home_team"]) | set(df["away_team"])
           if to_canonical(t) is None}
    check("all clubs canonical", not bad, f"{sorted(bad)}")

    # Home advantage should be visible. If it is not, something is wrong.
    home_win_rate = (df["result"] == "H").mean()
    check("home win rate between 0.40 and 0.50",
          0.40 <= home_win_rate <= 0.50, f"got {home_win_rate:.3f}")

    draw_rate = (df["result"] == "D").mean()
    check("draw rate between 0.20 and 0.30",
          0.20 <= draw_rate <= 0.30, f"got {draw_rate:.3f}")

    # Stats columns are missing in the oldest seasons for some fields.
    for col in ["home_shots", "home_sot", "home_corners"]:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            if pct > 1:
                warn(f"{col} missing", f"{pct:.1f}% of rows")

    return df


def validate_xg(matches: pd.DataFrame | None) -> None:
    path = INTERIM_DIR / "xg_understat.csv"
    print(f"\n{path.name}")
    if not path.exists():
        warn("file exists", "run scripts/03_fetch_xg.py (optional but recommended)")
        return

    xg = pd.read_csv(path, parse_dates=["date"])
    check("file exists", True)
    check("no nulls in xg", xg[["home_xg", "away_xg"]].isna().sum().sum() == 0)
    check("xg values plausible (0 to 8)",
          xg[["home_xg", "away_xg"]].max().max() < 8)
    check("home xg > away xg on average",
          xg["home_xg"].mean() > xg["away_xg"].mean(),
          f"{xg['home_xg'].mean():.2f} vs {xg['away_xg'].mean():.2f}")

    if matches is not None:
        merged = matches.merge(
            xg, on=["date", "home_team", "away_team"], how="inner"
        )
        eligible = matches[matches["season_start"] >= XG_FIRST_SEASON]
        rate = len(merged) / max(len(eligible), 1)
        check("xg joins to at least 95% of eligible matches",
              rate >= 0.95, f"joined {len(merged)}/{len(eligible)} = {rate:.1%}")
        if rate < 0.95:
            print("      -> almost always a team name or a date mismatch.")
            print("      -> run scripts/02_audit_team_names.py")


def validate_players() -> None:
    print("\nplayer stat tables")
    found = sorted(INTERIM_DIR.glob("players_*.csv"))
    if not found:
        warn("player files exist", "run scripts/04_fetch_players.py")
        return
    for path in found:
        df = pd.read_csv(path, low_memory=False)
        check(f"{path.name} not empty", len(df) > 0, f"{len(df)} rows")


def main() -> None:
    print("Validating collected data\n" + "=" * 40)
    matches = validate_matches()
    validate_xg(matches)
    validate_players()

    print("\n" + "=" * 40)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nFix these before building features.")
        sys.exit(1)

    print("All checks passed.")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s), review but not blocking.")


if __name__ == "__main__":
    main()
