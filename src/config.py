from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Seasons are identified by their STARTING year.
# 2006 means the 2006/07 season.
FIRST_SEASON = 2006
LAST_SEASON = 2025

# Understat has no expected goals data before 2014/15.
XG_FIRST_SEASON = 2014

# FBref only publishes advanced player stats (xG, progressive passes, carries) from 2017/18 onwards for the big five leagues.
FBREF_FIRST_SEASON = 2017

MATCHES_PER_SEASON = 380


def season_code(start_year: int) -> str:
    # 2006 -> '0607'. This is the code football-data.co.uk uses in its URLs.
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_codes(first: int = FIRST_SEASON, last: int = LAST_SEASON) -> list[str]:
    return [season_code(y) for y in range(first, last + 1)]


def season_label(start_year: int) -> str:
    # 2006 -> '2006/07'. This is the human readable label used in the data.
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def start_year_from_code(code: str) -> int:
    # '0607' -> 2006. Handles the 1990s codes too.
    yy = int(code[:2])
    return 2000 + yy if yy < 90 else 1900 + yy
