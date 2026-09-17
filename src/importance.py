from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR

MINUTES_WEIGHT = 0.6
CONTRIBUTION_WEIGHT = 0.4


def _find_column(df: pd.DataFrame, *patterns: str) -> str | None:
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for col in df.columns:
            if rx.fullmatch(str(col).strip()):
                return col
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for col in df.columns:
            if rx.search(str(col).strip()):
                return col
    return None


def inspect_columns(path=None) -> None:
    # Print the columns of the standard player table. Run this first.
    path = path or (INTERIM_DIR / "players_standard.csv")
    df = pd.read_csv(path, low_memory=False, nrows=5)
    print(f"{path.name}: {len(df.columns)} columns\n")
    for col in df.columns:
        print(f"  {col}")


def normalise_player_name(name: str) -> str:
    # Strip accents and punctuation so 'Ødegaard' and 'Odegaard' match. Used when the LLM gives back a name typed by a journalist.
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def build_importance(
    standard_path=None,
    out_path=None,
) -> pd.DataFrame:
    standard_path = standard_path or (INTERIM_DIR / "players_standard.csv")
    df = pd.read_csv(standard_path, low_memory=False)

    cols = {
        "player": _find_column(df, r"player", r".*player.*"),
        "team": _find_column(df, r"team_canonical", r"team", r"squad"),
        "season": _find_column(df, r"season_start"),
        "position": _find_column(df, r"pos", r".*pos.*"),
        "minutes": _find_column(df, r".*Min$", r".*minutes.*", r"Min"),
        "goals": _find_column(df, r".*Gls$", r".*goals.*", r"Gls"),
        "assists": _find_column(df, r".*Ast$", r".*assists.*", r"Ast"),
    }

    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise KeyError(
            f"Could not locate columns for: {missing}. "
            f"Run importance.inspect_columns() and adjust the patterns."
        )

    out = pd.DataFrame({
        "season_start": df[cols["season"]],
        "team": df[cols["team"]],
        "player": df[cols["player"]],
        "position": df[cols["position"]],
        "minutes": pd.to_numeric(df[cols["minutes"]], errors="coerce").fillna(0),
        "goals": pd.to_numeric(df[cols["goals"]], errors="coerce").fillna(0),
        "assists": pd.to_numeric(df[cols["assists"]], errors="coerce").fillna(0),
    })
    out = out.dropna(subset=["team", "player"])
    out["contributions"] = out["goals"] + out["assists"]

    # Scale within each squad and season. Comparing a Man City reserve to a Luton starter across the whole league would be meaningless.
    grp = out.groupby(["season_start", "team"])
    minutes_max = grp["minutes"].transform("max").replace(0, 1)
    contrib_max = grp["contributions"].transform("max").replace(0, 1)

    out["importance"] = (
        MINUTES_WEIGHT * (out["minutes"] / minutes_max)
        + CONTRIBUTION_WEIGHT * (out["contributions"] / contrib_max)
    ).clip(0, 1).round(3)

    out["player_key"] = out["player"].map(normalise_player_name)

    out = out.sort_values(
        ["season_start", "team", "importance"], ascending=[True, True, False]
    ).reset_index(drop=True)

    out_path = out_path or (PROCESSED_DIR / "player_importance.csv")
    out.to_csv(out_path, index=False)
    return out


def load_importance(path=None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / "player_importance.csv")
    return pd.read_csv(path)


def squad(importance: pd.DataFrame, team: str, season_start: int | None = None,
          top: int = 25) -> pd.DataFrame:
    # The most important players at a club, most recent season by default.
    sub = importance[importance["team"] == team]
    if sub.empty:
        return sub
    season_start = season_start or int(sub["season_start"].max())
    sub = sub[sub["season_start"] == season_start]
    return sub.nlargest(top, "importance")[
        ["player", "position", "minutes", "goals", "assists", "importance"]
    ].reset_index(drop=True)


def lookup(importance: pd.DataFrame, player_name: str, team: str | None = None,
           default: float = 0.25) -> float:
    # Importance for a player named in a news article.
    key = normalise_player_name(player_name)
    sub = importance
    if team:
        sub = sub[sub["team"] == team]
    hits = sub[sub["player_key"] == key]

    if hits.empty and " " in key:
        surname = key.rsplit(" ", 1)[-1]
        hits = sub[sub["player_key"].str.endswith(" " + surname, na=False)]

    if hits.empty:
        return default
    return float(hits.nlargest(1, "season_start")["importance"].iloc[0])
