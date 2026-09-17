import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FIRST_SEASON,
    INTERIM_DIR,
    LAST_SEASON,
    RAW_DIR,
    season_code,
    season_label,
)
from src.teams import to_canonical, unmapped

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (student football-ml project)"}

# Source column, the name we actually want to work with.
RENAME = {
    "Date": "date_raw",
    "Time": "kickoff",
    "HomeTeam": "home_raw",
    "AwayTeam": "away_raw",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "home_ht_goals",
    "HTAG": "away_ht_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}


def download(start_year: int, force: bool = False) -> Path:
    code = season_code(start_year)
    dest = RAW_DIR / f"E0_{code}.csv"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  cached   {dest.name}")
        return dest
    url = URL.format(code=code)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  fetched  {dest.name}  ({len(resp.content) // 1024} KB)")
    time.sleep(1.0)  # be polite, it is a free service
    return dest


def load_season(path: Path, start_year: int) -> pd.DataFrame:
    # latin-1 because these files are not consistently UTF-8
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])

    keep = [c for c in RENAME if c in df.columns]
    df = df[keep].rename(columns=RENAME)

    df["season_start"] = start_year
    df["season"] = season_label(start_year)
    return df


def main(force: bool = False) -> None:
    print(f"Downloading seasons {FIRST_SEASON}/{FIRST_SEASON % 100 + 1} "
          f"to {LAST_SEASON}/{(LAST_SEASON + 1) % 100}")

    frames = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        path = download(year, force=force)
        frames.append(load_season(path, year))

    df = pd.concat(frames, ignore_index=True)
    print(f"\nRaw rows: {len(df)}")

    # Dates are dd/mm/yy in old seasons and dd/mm/yyyy in newer ones.
    df["date"] = pd.to_datetime(
        df["date_raw"], dayfirst=True, format="mixed", errors="coerce"
    )
    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        print(f"WARNING: {bad_dates} rows had an unparseable date, dropping them")
        df = df.dropna(subset=["date"])

    # Canonical club names.
    df["home_team"] = df["home_raw"].map(to_canonical)
    df["away_team"] = df["away_raw"].map(to_canonical)

    missing = unmapped(
        set(df.loc[df["home_team"].isna(), "home_raw"])
        | set(df.loc[df["away_team"].isna(), "away_raw"])
    )
    if missing:
        print("\n!! Unmapped club names. Add these to src/teams.py and re-run:")
        for name in missing:
            print(f"     {name!r}")
        sys.exit(1)

    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "match_id", range(1, len(df) + 1))

    ordered = [
        "match_id", "date", "season", "season_start",
        "home_team", "away_team",
        "home_goals", "away_goals", "result",
        "home_ht_goals", "away_ht_goals",
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellow", "away_yellow", "home_red", "away_red",
        "odds_home", "odds_draw", "odds_away",
    ]
    df = df[[c for c in ordered if c in df.columns]]

    out = INTERIM_DIR / "matches_football_data.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {len(df)} matches to {out}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Seasons:    {df['season'].nunique()}")
    print(f"Clubs:      {df['home_team'].nunique()}")
    print("\nResult split:")
    print((df["result"].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    main(**vars(ap.parse_args()))
