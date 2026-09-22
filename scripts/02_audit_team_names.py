import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from src.teams import canonical_clubs, to_canonical

# Column names that hold a club name in one source or another.
TEAM_COLUMNS = {
    "home_raw", "away_raw", "home_team", "away_team",
    "team", "Team", "squad", "Squad", "home", "away",
    "home_team_name", "away_team_name",
}


def collect_names() -> dict[str, set[str]]:
    # Return {file: {unresolvable names}} across every csv we have.
    found: dict[str, set[str]] = {}
    roots = [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]

    for root in roots:
        for path in sorted(root.rglob("*.csv")):
            try:
                df = pd.read_csv(path, encoding="latin-1", nrows=200_000,
                                 on_bad_lines="skip", low_memory=False)
            except Exception as exc:
                print(f"  skipped {path.name}: {exc}")
                continue

            cols = [c for c in df.columns if c in TEAM_COLUMNS]
            if not cols:
                continue

            names = set()
            for col in cols:
                names |= {str(v) for v in df[col].dropna().unique()}

            bad = {n for n in names if to_canonical(n) is None}
            if bad:
                found[str(path.relative_to(path.parents[2]))] = bad

    return found


def main() -> None:
    canon = canonical_clubs()
    found = collect_names()

    if not found:
        print("All club names resolve cleanly. Safe to merge.")
        return

    print("Unresolved club names found.\n")
    print("Add each to the right entry in CLUB_ALIASES in src/teams.py.\n")

    for file, names in found.items():
        print(f"{file}")
        for name in sorted(names):
            match = process.extractOne(name, canon)
            hint = f"  ->  did you mean {match[0]!r}? (score {match[1]:.0f})" if match else ""
            print(f"    {name!r}{hint}")
        print()

    sys.exit(1)


if __name__ == "__main__":
    main()
