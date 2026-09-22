import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR

FULL_DIR = RAW_DIR / "transfermarkt"
PL_DIR = RAW_DIR / "transfermarkt_pl"
PL_DIR.mkdir(parents=True, exist_ok=True)

PREMIER_LEAGUE = "GB1"
GITHUB_WARN_MB = 50


def read_full(name: str) -> pd.DataFrame:
    for ext in (".csv.gz", ".csv"):
        path = FULL_DIR / f"{name}{ext}"
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(
        f"{name} not found in {FULL_DIR}. "
        "Run scripts/06_fetch_transfermarkt.py first."
    )


def write(df: pd.DataFrame, name: str) -> float:
    dest = PL_DIR / f"{name}.csv.gz"
    df.to_csv(dest, index=False, compression="gzip")
    mb = dest.stat().st_size / 1024 / 1024
    flag = "  <- over the GitHub warning size" if mb > GITHUB_WARN_MB else ""
    print(f"  {dest.name:32} {len(df):>10,} rows   {mb:6.1f} MB{flag}")
    return mb


def main() -> None:
    print("Reading the full tables. appearances is large, give it a moment.\n")

    appearances = read_full("appearances")
    pl_apps = appearances[appearances["competition_id"] == PREMIER_LEAGUE]
    pl_players = set(pl_apps["player_id"].unique())

    print(f"appearances: {len(appearances):,} total, "
          f"{len(pl_apps):,} in the Premier League")
    print(f"players who appeared in the PL: {len(pl_players):,}\n")

    players = read_full("players")
    valuations = read_full("player_valuations")
    clubs = read_full("clubs")
    competitions = read_full("competitions")

    print(f"Writing extract to {PL_DIR}\n")
    total = 0.0
    total += write(pl_apps, "appearances")
    total += write(players[players["player_id"].isin(pl_players)], "players")
    total += write(valuations[valuations["player_id"].isin(pl_players)],
                   "player_valuations")
    total += write(clubs, "clubs")
    total += write(competitions, "competitions")

    print(f"\n  total {total:.1f} MB")
    print("\nCommit data/raw/transfermarkt_pl/.")
    print("Keep data/raw/transfermarkt/ in .gitignore, it stays on your machine.")


if __name__ == "__main__":
    main()
