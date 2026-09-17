import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import INTERIM_DIR, LAST_SEASON, RAW_DIR, XG_FIRST_SEASON, season_code
from src.teams import to_canonical, unmapped

# Possible column names for each field. The script picks the first one that actually exists, so a rename upstream does not break everything.
CANDIDATES = {
    "date": ["date", "datetime", "game_date"],
    "home_raw": ["home_team", "home", "team_home"],
    "away_raw": ["away_team", "away", "team_away"],
    "home_xg": ["home_xg", "xg_home", "home_expected_goals"],
    "away_xg": ["away_xg", "xg_away", "away_expected_goals"],
}


def pick(df: pd.DataFrame, options: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt in lower:
            return lower[opt]
    return None


def fetch() -> pd.DataFrame:
    import soccerdata as sd

    seasons = [season_code(y) for y in range(XG_FIRST_SEASON, LAST_SEASON + 1)]
    print(f"Fetching Understat for {len(seasons)} seasons: {seasons[0]} to {seasons[-1]}")

    understat = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
    df = understat.read_schedule().reset_index()
    return df


def main(inspect: bool = False) -> None:
    df = fetch()

    raw_out = RAW_DIR / "understat_schedule_raw.csv"
    df.to_csv(raw_out, index=False)
    print(f"Saved raw Understat table to {raw_out}  ({len(df)} rows)")

    if inspect:
        print("\nColumns returned by soccerdata:")
        for col in df.columns:
            print(f"  {col}")
        print("\nFirst two rows:")
        print(df.head(2).to_string())
        print("\nIf the xG columns are not named as CANDIDATES expects, "
              "edit CANDIDATES at the top of this script.")
        return

    mapping = {}
    for target, options in CANDIDATES.items():
        found = pick(df, options)
        if found is None:
            print(f"\nERROR: could not find a column for {target!r}.")
            print("Run with --inspect and update CANDIDATES.")
            sys.exit(1)
        mapping[found] = target

    out = df[list(mapping)].rename(columns=mapping).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "home_xg", "away_xg"])

    out["home_team"] = out["home_raw"].map(to_canonical)
    out["away_team"] = out["away_raw"].map(to_canonical)

    missing = unmapped(
        set(out.loc[out["home_team"].isna(), "home_raw"])
        | set(out.loc[out["away_team"].isna(), "away_raw"])
    )
    if missing:
        print("\n!! Unmapped club names from Understat. Add to src/teams.py:")
        for name in missing:
            print(f"     {name!r}")
        sys.exit(1)

    out = out[["date", "home_team", "away_team", "home_xg", "away_xg"]]
    out = out.sort_values("date").reset_index(drop=True)

    dest = INTERIM_DIR / "xg_understat.csv"
    out.to_csv(dest, index=False)

    print(f"\nWrote {len(out)} matches with xG to {dest}")
    print(f"Date range: {out['date'].min().date()} to {out['date'].max().date()}")
    print(f"Mean home xG: {out['home_xg'].mean():.2f}   "
          f"mean away xG: {out['away_xg'].mean():.2f}")
    print("\nSanity check: home xG should be clearly higher than away xG.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="print the columns and exit without normalising")
    main(**vars(ap.parse_args()))
