from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from src.teams import to_canonical

# The PL-only extract is committed to the repo. The full dataset stays local. Reading the extract first means a fresh clone just works.
TM_DIRS = [RAW_DIR / "transfermarkt_pl", RAW_DIR / "transfermarkt"]
TM_DIR = TM_DIRS[-1]
PREMIER_LEAGUE = "GB1"

# Season runs August to May. A match on 2024-09-14 belongs to 2024/25, one on 2025-03-02 also belongs to 2024/25.
SEASON_CUTOFF_MONTH = 7


def _read(name: str, **kw) -> pd.DataFrame:
    for folder in TM_DIRS:
        for ext in (".csv.gz", ".csv"):
            path = folder / f"{name}{ext}"
            if path.exists() and path.stat().st_size > 0:
                return pd.read_csv(path, low_memory=False, **kw)
    raise FileNotFoundError(
        f"{name} not found in {[str(d) for d in TM_DIRS]}. "
        "Run scripts/06_check_transfermarkt.py, then 07_trim_transfermarkt.py"
    )


def season_of(dates: pd.Series) -> pd.Series:
    # Map a date to the starting year of its season.
    d = pd.to_datetime(dates, errors="coerce")
    return np.where(d.dt.month > SEASON_CUTOFF_MONTH, d.dt.year, d.dt.year - 1)


def build_player_seasons(first_season: int = 2012) -> pd.DataFrame:
    appearances = _read("appearances")
    players = _read("players")
    valuations = _read("player_valuations")
    clubs = _read("clubs")

    # 1. Premier League appearances only 
    apps = appearances[appearances["competition_id"] == PREMIER_LEAGUE].copy()
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps["season_start"] = season_of(apps["date"])
    apps = apps[apps["season_start"] >= first_season]

    # 2. Aggregate a season of output per player 
    agg = apps.groupby(["player_id", "season_start"]).agg(
        appearances=("game_id", "count"),
        minutes=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        season_end=("date", "max"),
    ).reset_index()

    # Which club did they play most of the season for? Loan moves and January transfers mean a player can appear for two.
    main_club = (
        apps.groupby(["player_id", "season_start", "player_club_id"])["minutes_played"]
        .sum().reset_index()
        .sort_values("minutes_played", ascending=False)
        .drop_duplicates(["player_id", "season_start"])
        .rename(columns={"player_club_id": "club_id"})
        .drop(columns="minutes_played")
    )
    df = agg.merge(main_club, on=["player_id", "season_start"], how="left")

    # 3. Club names, mapped to our canonical set 
    club_cols = [c for c in ("club_id", "name", "club_code") if c in clubs.columns]
    df = df.merge(clubs[club_cols], on="club_id", how="left")
    df["club"] = df["name"].map(to_canonical)
    df = df.drop(columns=[c for c in ("name", "club_code") if c in df.columns])

    # 4. Player attributes 
    keep = ["player_id", "name", "date_of_birth", "position", "sub_position",
            "foot", "height_in_cm", "country_of_citizenship",
            "contract_expiration_date"]
    keep = [c for c in keep if c in players.columns]
    df = df.merge(players[keep], on="player_id", how="left")
    df = df.rename(columns={"name": "player"})

    # 5. The target: valuation at the end of that season 
    val = valuations.copy()
    val["date"] = pd.to_datetime(val["date"], errors="coerce")
    val = val.dropna(subset=["date", "market_value_in_eur"])

    df = _attach_valuation(df, val, column="market_value_eur", offset_days=0)
    # The same player's value a year earlier. Powerful and dangerous, see the note in add_features.
    df = _attach_valuation(df, val, column="market_value_prev_eur",
                           offset_days=365)

    df = df.dropna(subset=["market_value_eur"])
    return df.sort_values(["season_start", "player"]).reset_index(drop=True)


def _attach_valuation(df: pd.DataFrame, val: pd.DataFrame, column: str,
                      offset_days: int) -> pd.DataFrame:
    target = df[["player_id", "season_start", "season_end"]].copy()
    target["cutoff"] = pd.to_datetime(target["season_end"]) - pd.Timedelta(
        days=offset_days
    )
    target = target.sort_values("cutoff")

    v = val[["player_id", "date", "market_value_in_eur"]].sort_values("date")

    merged = pd.merge_asof(
        target, v,
        left_on="cutoff", right_on="date",
        by="player_id",
        direction="backward",
        tolerance=pd.Timedelta(days=400),
    )
    merged = merged.rename(columns={"market_value_in_eur": column})

    return df.merge(
        merged[["player_id", "season_start", column]],
        on=["player_id", "season_start"], how="left",
    )


def add_features(df: pd.DataFrame, club_elo: pd.DataFrame | None = None,
                 fbref: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()

    # Age at the end of the season. This is the single strongest predictor and it is not linear: value peaks around 24 to 26 and falls off steeply after 29, so the square gets its own column.
    dob = pd.to_datetime(out["date_of_birth"], errors="coerce")
    end = pd.to_datetime(out["season_end"], errors="coerce")
    out["age"] = (end - dob).dt.days / 365.25
    out["age_squared"] = out["age"] ** 2

    # Contract security. A player with six months left is worth less, because any buyer can wait.
    expiry = pd.to_datetime(out["contract_expiration_date"], errors="coerce")
    out["contract_years_left"] = (expiry - end).dt.days / 365.25
    out["contract_years_left"] = out["contract_years_left"].clip(-1, 6)

    # Output. Per 90 matters more than totals: 10 goals in 900 minutes is a different player from 10 goals in 3,000.
    minutes = out["minutes"].clip(lower=1)
    out["nineties"] = out["minutes"] / 90.0
    for stat in ("goals", "assists"):
        out[f"{stat}_per90"] = out[stat] / (minutes / 90.0)
    out["contributions"] = out["goals"] + out["assists"]
    out["contributions_per90"] = out["contributions"] / (minutes / 90.0)
    out["minutes_per_app"] = out["minutes"] / out["appearances"].clip(lower=1)

    # Playing regularly is itself a signal of quality.
    out["minutes_share"] = (out["minutes"] / (38 * 90)).clip(0, 1)

    # Club strength.
    if club_elo is not None:
        elo = club_elo.rename(columns={"team": "club"})[["club", "elo"]]
        out = out.merge(elo, on="club", how="left")
        out["elo"] = out["elo"].fillna(out["elo"].median())
    else:
        out["elo"] = np.nan

    # Optional xG from FBref, joined on name and club and season.
    if fbref is not None:
        out = _join_fbref(out, fbref)

    # Position, collapsed to the four that matter for valuation.
    out["pos_group"] = out["position"].map({
        "Goalkeeper": "GK", "Defender": "DF",
        "Midfield": "MF", "Attack": "FW",
    }).fillna("MF")

    out["log_value"] = np.log(out["market_value_eur"])
    out["log_value_prev"] = np.log(out["market_value_prev_eur"].replace(0, np.nan))

    return out


def _join_fbref(df: pd.DataFrame, fbref: pd.DataFrame) -> pd.DataFrame:
    from src.importance import normalise_player_name

    f = fbref.copy()
    name_col = next((c for c in ("player", "Player") if c in f.columns), None)
    if name_col is None:
        return df

    xg_col = next((c for c in f.columns if str(c).lower().endswith("_xg")
                   or str(c).lower() == "xg"), None)
    if xg_col is None:
        return df

    f["_key"] = f[name_col].map(normalise_player_name)
    f = f.groupby(["_key", "season_start"], as_index=False)[xg_col].max()
    f = f.rename(columns={xg_col: "xg"})

    df = df.copy()
    df["_key"] = df["player"].map(normalise_player_name)
    out = df.merge(f, on=["_key", "season_start"], how="left").drop(columns="_key")

    matched = out["xg"].notna().mean()
    print(f"  xG joined to {matched:.1%} of rows")
    return out


FEATURES_CORE = [
    "age", "age_squared", "contract_years_left",
    "minutes", "nineties", "appearances", "minutes_per_app", "minutes_share",
    "goals", "assists", "contributions",
    "goals_per90", "assists_per90", "contributions_per90",
    "yellow_cards", "red_cards", "height_in_cm", "elo",
]


def feature_matrix(df: pd.DataFrame, include_prev_value: bool = False,
                   include_xg: bool = True) -> tuple[pd.DataFrame, list[str]]:
    cols = list(FEATURES_CORE)
    if include_xg and "xg" in df.columns:
        cols.append("xg")
    if include_prev_value and "log_value_prev" in df.columns:
        cols.append("log_value_prev")

    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()

    # Position and foot as dummies.
    for cat in ("pos_group", "foot"):
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat].fillna("unknown"), prefix=cat,
                                     dtype=float)
            X = pd.concat([X, dummies], axis=1)

    return X, list(X.columns)


def build(first_season: int = 2012, club_elo=None, fbref=None,
          save: bool = True) -> pd.DataFrame:
    # Full pipeline. Returns the feature table and optionally saves it.
    df = build_player_seasons(first_season=first_season)
    df = add_features(df, club_elo=club_elo, fbref=fbref)
    if save:
        df.to_csv(PROCESSED_DIR / "player_market_values.csv", index=False)
    return df


def euros(x) -> str:
    # Format a value the way a person would say it.
    x = float(x)
    if x >= 1e6:
        return f"EUR {x / 1e6:.1f}m"
    if x >= 1e3:
        return f"EUR {x / 1e3:.0f}k"
    return f"EUR {x:.0f}"


def error_report(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    true = np.exp(y_true_log)
    pred = np.exp(y_pred_log)
    ape = np.abs(pred - true) / true

    return {
        "median_ape": float(np.median(ape)),
        "mean_ape": float(np.mean(ape)),
        "median_abs_error_eur": float(np.median(np.abs(pred - true))),
        "within_25pct": float(np.mean(ape <= 0.25)),
        "within_50pct": float(np.mean(ape <= 0.50)),
        "log_rmse": float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))),
        "log_r2": float(1 - np.var(y_pred_log - y_true_log) / np.var(y_true_log)),
    }
