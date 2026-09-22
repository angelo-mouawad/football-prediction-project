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
        "Run scripts/06_fetch_transfermarkt.py, then 07_trim_transfermarkt.py"
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

    # Club strength, matched to the season the player was there.
    # Pass features.season_elo_table(...) here. If you pass a table with no season column it falls back to one rating per club, which is wrong for historical seasons and kept only for compatibility.
    if club_elo is not None:
        elo = club_elo.rename(columns={"team": "club"})
        keys = ["club", "season_start"] if "season_start" in elo.columns else ["club"]
        out = out.merge(elo[keys + ["elo"]], on=keys, how="left")
        season_median = out.groupby("season_start")["elo"].transform("median")
        out["elo"] = out["elo"].fillna(season_median).fillna(out["elo"].median())
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


def _find_xg_column(frame: pd.DataFrame) -> str | None:
    cols = [str(c) for c in frame.columns]
    exact = [c for c in cols if c.lower() in ("xg", "expected_xg")
             or c.lower().endswith("_xg")]
    if exact:
        return exact[0]
    loose = [c for c in cols if "xg" in c.lower()
             and not any(bad in c.lower() for bad in ("npxg", "xag", "xa_", "per"))]
    return loose[0] if loose else None


def _join_fbref(df: pd.DataFrame, fbref) -> pd.DataFrame:
    from src.importance import normalise_player_name

    tables = fbref if isinstance(fbref, (list, tuple)) else [fbref]

    for f in tables:
        if f is None:
            continue
        name_col = next((c for c in ("player", "Player") if c in f.columns), None)
        xg_col = _find_xg_column(f)
        if name_col is None or xg_col is None or "season_start" not in f.columns:
            continue

        f = f.copy()
        f["_key"] = f[name_col].map(normalise_player_name)
        f[xg_col] = pd.to_numeric(f[xg_col], errors="coerce")
        f = f.groupby(["_key", "season_start"], as_index=False)[xg_col].max()
        f = f.rename(columns={xg_col: "xg"})

        out = df.copy()
        out["_key"] = out["player"].map(normalise_player_name)
        out = out.merge(f, on=["_key", "season_start"], how="left").drop(columns="_key")
        print(f"  xG joined from column {xg_col!r} to {out['xg'].notna().mean():.1%} of rows")
        return out

    print("  No xG column in any FBref table. Continuing without xG.")
    print("  Your FBref scrape only offered basic stat types, so this is expected.")
    return df


FEATURES_CORE = [
    "age", "age_squared", "contract_years_left",
    "minutes", "nineties", "appearances", "minutes_per_app", "minutes_share",
    "goals", "assists", "contributions",
    "goals_per90", "assists_per90", "contributions_per90",
    "yellow_cards", "red_cards", "height_in_cm", "elo",
]


def season_index(df: pd.DataFrame, col: str = "log_value") -> pd.Series:
    # Median log value per season. The price level of the market.
    return df.groupby("season_start")[col].median()


def index_for(seasons, index: pd.Series) -> np.ndarray:
    x = index.index.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, index.to_numpy(dtype=float), 1)
    seasons = np.asarray(seasons)
    return np.array([
        index[s] if s in index.index else slope * s + intercept
        for s in seasons
    ], dtype=float)


def add_relative_target(df: pd.DataFrame, known_seasons) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    idx = season_index(out[out["season_start"].isin(list(known_seasons))])

    out["market_index"] = index_for(out["season_start"], idx)
    out["log_value_rel"] = out["log_value"] - out["market_index"]

    prev_index = index_for(out["season_start"] - 1, idx)
    out["log_value_prev_rel"] = out["log_value_prev"] - prev_index
    return out, idx


def feature_matrix(df: pd.DataFrame, include_prev_value: bool = False,
                   include_xg: bool = True) -> tuple[pd.DataFrame, list[str]]:
    cols = list(FEATURES_CORE)
    if include_xg and "xg" in df.columns:
        cols.append("xg")
    if include_prev_value:
        prev = "log_value_prev_rel" if "log_value_prev_rel" in df.columns \
            else "log_value_prev"
        if prev in df.columns:
            cols.append(prev)

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
