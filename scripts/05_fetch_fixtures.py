import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.teams import to_canonical, unmapped

load_dotenv(PROJECT_ROOT / ".env")

import os

API = "https://api.football-data.org/v4/competitions/PL/matches"


def main() -> None:
    key = os.getenv("FOOTBALL_DATA_ORG_KEY", "").strip()
    if not key:
        print("No FOOTBALL_DATA_ORG_KEY in .env")
        print("Register free at https://www.football-data.org/client/register")
        sys.exit(1)

    resp = requests.get(
        API,
        headers={"X-Auth-Token": key},
        params={"status": "SCHEDULED"},
        timeout=30,
    )
    if resp.status_code == 403:
        print("403 from the API. Free tier may not cover this competition, "
              "or the key is wrong.")
        sys.exit(1)
    resp.raise_for_status()

    matches = resp.json().get("matches", [])
    if not matches:
        print("No scheduled matches returned. Off season, or all fixtures played.")
        return

    rows = [
        {
            "fixture_id": m["id"],
            "utc_date": m["utcDate"],
            "matchday": m.get("matchday"),
            "home_raw": m["homeTeam"]["name"],
            "away_raw": m["awayTeam"]["name"],
        }
        for m in matches
    ]
    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["utc_date"], errors="coerce").dt.tz_localize(None)
    df["home_team"] = df["home_raw"].map(to_canonical)
    df["away_team"] = df["away_raw"].map(to_canonical)

    missing = unmapped(
        set(df.loc[df["home_team"].isna(), "home_raw"])
        | set(df.loc[df["away_team"].isna(), "away_raw"])
    )
    if missing:
        print("\n!! Unmapped club names from football-data.org. Add to src/teams.py:")
        for name in missing:
            print(f"     {name!r}")
        sys.exit(1)

    df = df[["fixture_id", "date", "matchday", "home_team", "away_team"]]
    df = df.sort_values("date").reset_index(drop=True)

    dest = PROCESSED_DIR / "fixtures_upcoming.csv"
    df.to_csv(dest, index=False)

    print(f"Wrote {len(df)} upcoming fixtures to {dest}")
    print("\nNext five:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
