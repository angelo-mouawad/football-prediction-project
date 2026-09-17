import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FBREF_FIRST_SEASON,
    INTERIM_DIR,
    LAST_SEASON,
    RAW_DIR,
    season_code,
    season_label,
)
from src.teams import to_canonical

STAT_TYPES = ["standard", "keeper", "shooting", "playing_time", "misc"]

FBREF_DIR = RAW_DIR / "fbref"
FBREF_DIR.mkdir(parents=True, exist_ok=True)


def chunk_path(year: int, stat: str) -> Path:
    return FBREF_DIR / f"players_{season_code(year)}_{stat}.csv"


def fetch_chunk(year: int, stat: str, sleep: float) -> pd.DataFrame | None:
    import soccerdata as sd

    dest = chunk_path(year, stat)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached   {dest.name}")
        return pd.read_csv(dest, low_memory=False)

    print(f"  fetching {dest.name} ...", flush=True)
    try:
        fbref = sd.FBref(leagues="ENG-Premier League", seasons=[season_code(year)])
        df = fbref.read_player_season_stats(stat_type=stat)
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return None

    # soccerdata returns a MultiIndex on both axes. Flatten both.
    df = df.reset_index()
    df.columns = [
        "_".join(str(p) for p in col if p and not str(p).startswith("Unnamed"))
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]

    df["season_start"] = year
    df["season"] = season_label(year)
    df["stat_type"] = stat
    df.to_csv(dest, index=False)

    print(f"    saved {len(df)} rows, {len(df.columns)} columns")
    time.sleep(sleep)
    return df


def main(sleep: float = 6.0, first: int = FBREF_FIRST_SEASON) -> None:
    years = list(range(first, LAST_SEASON + 1))
    print(f"FBref player stats: {len(years)} seasons x {len(STAT_TYPES)} stat types")
    print(f"Sleeping {sleep}s between requests. This will take a while.\n")

    per_stat: dict[str, list[pd.DataFrame]] = {s: [] for s in STAT_TYPES}

    for year in years:
        print(f"{season_label(year)}")
        for stat in STAT_TYPES:
            df = fetch_chunk(year, stat, sleep)
            if df is not None:
                per_stat[stat].append(df)

    print("\nCombining ...")
    for stat, frames in per_stat.items():
        if not frames:
            print(f"  {stat}: nothing collected, skipping")
            continue
        combined = pd.concat(frames, ignore_index=True)

        team_col = next(
            (c for c in ("team", "Team", "squad", "Squad") if c in combined.columns),
            None,
        )
        if team_col:
            combined["team_canonical"] = combined[team_col].map(to_canonical)
            bad = combined["team_canonical"].isna().sum()
            if bad:
                names = sorted(combined.loc[combined["team_canonical"].isna(), team_col]
                               .dropna().unique())
                print(f"  {stat}: {bad} rows with unmapped clubs -> {names[:10]}")

        dest = INTERIM_DIR / f"players_{stat}.csv"
        combined.to_csv(dest, index=False)
        print(f"  {stat}: {len(combined)} rows -> {dest.name}")

    print("\nDone. Now run: python scripts/02_audit_team_names.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=6.0,
                    help="seconds between FBref requests (raise if blocked)")
    ap.add_argument("--first", type=int, default=FBREF_FIRST_SEASON,
                    help="first season start year to fetch")
    main(**vars(ap.parse_args()))
