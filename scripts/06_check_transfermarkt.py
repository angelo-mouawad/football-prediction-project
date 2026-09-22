import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR

FOLDERS = [RAW_DIR / "transfermarkt_pl", RAW_DIR / "transfermarkt"]

NEEDED = [
    "players",
    "player_valuations",
    "appearances",
    "clubs",
    "competitions",
]


def find(name: str) -> Path | None:
    for folder in FOLDERS:
        for ext in (".csv.gz", ".csv"):
            path = folder / f"{name}{ext}"
            if path.exists() and path.stat().st_size > 0:
                return path
    return None


def main() -> None:
    missing = []

    for name in NEEDED:
        path = find(name)
        if path is None:
            print(f"  missing  {name}")
            missing.append(name)
            continue

        try:
            head = pd.read_csv(path, nrows=3, low_memory=False)
        except Exception as exc:
            print(f"  broken   {path.parent.name}/{path.name}: {exc}")
            missing.append(name)
            continue

        mb = path.stat().st_size / 1024 / 1024
        print(f"  ok       {path.parent.name}/{path.name}  "
              f"({mb:.1f} MB, {len(head.columns)} columns)")

    print()
    if missing:
        print(f"Missing {len(missing)} of {len(NEEDED)} tables.")
        print("Download them from Kaggle (davidcariboo/player-scores) and")
        print(f"extract them into {FOLDERS[-1]}")
        sys.exit(1)

    print("All tables present.")


if __name__ == "__main__":
    main()
