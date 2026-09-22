from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Elo settings. These are conventional football values, not tuned.
ELO_START = 1500.0
ELO_PROMOTED = 1400.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0
ELO_SEASON_REGRESSION = 0.25

POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def _expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    start: float = ELO_START,
    promoted: float = ELO_PROMOTED,
    regression: float = ELO_SEASON_REGRESSION,
) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True).copy()

    ratings: dict[str, float] = {}
    first_season = df["season_start"].min()
    current_season = None
    seen_this_season: set[str] = set()
    debut_season: dict[str, int] = {}

    elo_h, elo_a = [], []
    promo_h, promo_a = [], []

    for row in df.itertuples(index=False):
        # Between seasons, drag everyone toward the mean. Squads change.
        if row.season_start != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] += (start - ratings[team]) * regression
            current_season = row.season_start
            seen_this_season = set()

        for team in (row.home_team, row.away_team):
            if team not in ratings:
                ratings[team] = start if row.season_start == first_season else promoted
                debut_season[team] = row.season_start
            seen_this_season.add(team)

        rh, ra = ratings[row.home_team], ratings[row.away_team]
        elo_h.append(rh)
        elo_a.append(ra)

        # A club is "promoted" for its first season back in the league.
        promo_h.append(int(debut_season[row.home_team] == row.season_start
                           and row.season_start != first_season))
        promo_a.append(int(debut_season[row.away_team] == row.season_start
                           and row.season_start != first_season))

        # Update. Margin of victory scales K: a 4-0 says more than a 1-0.
        exp_h = _expected(rh + home_adv, ra)
        score_h = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        gd = abs(row.home_goals - row.away_goals)
        k_eff = k * (1.0 + 0.5 * math.log1p(gd))

        delta = k_eff * (score_h - exp_h)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["is_promoted_home"] = promo_h
    df["is_promoted_away"] = promo_a
    return df


def final_elo_table(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    # Current Elo per club, for the dashboard and for sanity checking.
    rows = []
    for _, r in matches_with_elo.iterrows():
        rows.append((r["date"], r["home_team"], r["elo_home"]))
        rows.append((r["date"], r["away_team"], r["elo_away"]))
    long = pd.DataFrame(rows, columns=["date", "team", "elo"])
    latest = long.sort_values("date").groupby("team").tail(1)
    return latest.sort_values("elo", ascending=False).reset_index(drop=True)


def to_team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    base = ["match_id", "date", "season_start"]

    home = matches[base].copy()
    home["team"] = matches["home_team"]
    home["opponent"] = matches["away_team"]
    home["is_home"] = 1
    home["goals_for"] = matches["home_goals"]
    home["goals_against"] = matches["away_goals"]
    home["points"] = matches["result"].map(lambda r: POINTS[r][0])

    away = matches[base].copy()
    away["team"] = matches["away_team"]
    away["opponent"] = matches["home_team"]
    away["is_home"] = 0
    away["goals_for"] = matches["away_goals"]
    away["goals_against"] = matches["home_goals"]
    away["points"] = matches["result"].map(lambda r: POINTS[r][1])

    # Optional per-match stats, carried across when present.
    pairs = {
        "shots": ("home_shots", "away_shots"),
        "shots_against": ("away_shots", "home_shots"),
        "sot": ("home_sot", "away_sot"),
        "sot_against": ("away_sot", "home_sot"),
        "corners": ("home_corners", "away_corners"),
        "xg": ("home_xg", "away_xg"),
        "xg_against": ("away_xg", "home_xg"),
    }
    for name, (hcol, acol) in pairs.items():
        if hcol in matches.columns and acol in matches.columns:
            home[name] = matches[hcol]
            away[name] = matches[acol]

    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _shifted_rolling(group: pd.Series, window: int) -> pd.Series:
    # Mean of the previous `window` values, excluding the current one.
    return group.shift(1).rolling(window, min_periods=1).mean()


def add_rolling_features(
    long: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    df = long.sort_values(["team", "date", "match_id"]).copy()

    stats = ["points", "goals_for", "goals_against"]
    for optional in ("shots", "sot", "shots_against", "sot_against", "corners",
                     "xg", "xg_against"):
        if optional in df.columns:
            stats.append(optional)

    grouped = df.groupby("team", sort=False)
    for w in windows:
        for stat in stats:
            df[f"{stat}_last{w}"] = grouped[stat].transform(
                lambda s, w=w: _shifted_rolling(s, w)
            )

    # Venue specific form. Home form at home, away form away.
    venue = df.groupby(["team", "is_home"], sort=False)
    df["points_venue_last5"] = venue["points"].transform(
        lambda s: _shifted_rolling(s, 5)
    )

    # Days since the team last played.
    df["rest_days"] = grouped["date"].diff().dt.days

    # Fixture congestion: matches played in the previous 14 days.
    counts = []
    for _, sub in df.groupby("team", sort=False):
        s = pd.Series(1.0, index=pd.DatetimeIndex(sub["date"]))
        rolled = s.rolling("14D").sum().to_numpy() - 1.0
        counts.append(pd.Series(rolled, index=sub.index))
    df["matches_last14"] = pd.concat(counts).sort_index()

    # How far into the season we are. Early season form is thin.
    df["matchweek"] = df.groupby(["team", "season_start"], sort=False).cumcount() + 1

    return df


def add_head_to_head(matches: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True).copy()
    history: dict[frozenset, list[tuple[str, int]]] = {}
    values = []

    for row in df.itertuples(index=False):
        pair = frozenset((row.home_team, row.away_team))
        past = history.get(pair, [])
        if past:
            recent = past[-n:]
            pts = [p for team, p in recent if team == row.home_team]
            values.append(float(np.mean(pts)) if pts else np.nan)
        else:
            values.append(np.nan)

        hp, ap = POINTS[row.result]
        history.setdefault(pair, []).extend(
            [(row.home_team, hp), (row.away_team, ap)]
        )

    df["h2h_home_ppg"] = values
    return df


FEATURE_SUFFIXES = [
    "points_last5", "points_last10",
    "goals_for_last5", "goals_for_last10",
    "goals_against_last5", "goals_against_last10",
    "shots_last5", "sot_last5",
    "shots_against_last5", "sot_against_last5",
    "corners_last5",
    "xg_last5", "xg_last10",
    "xg_against_last5", "xg_against_last10",
    "points_venue_last5",
    "rest_days", "matches_last14", "matchweek",
]


def build_dataset(
    matches: pd.DataFrame,
    xg: pd.DataFrame | None = None,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])

    if xg is not None and len(xg):
        x = xg.copy()
        x["date"] = pd.to_datetime(x["date"]).dt.normalize()
        df["_d"] = df["date"].dt.normalize()
        before = len(df)
        df = df.merge(
            x[["date", "home_team", "away_team", "home_xg", "away_xg"]],
            left_on=["_d", "home_team", "away_team"],
            right_on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_xg"),
        ).drop(columns=["_d", "date_xg"], errors="ignore")
        assert len(df) == before, "xG merge duplicated rows, check for repeat fixtures"

    df = compute_elo(df)
    df = add_head_to_head(df)

    long = to_team_match_long(df)
    long = add_rolling_features(long, windows=windows)

    keep = ["match_id", "team"] + [c for c in FEATURE_SUFFIXES if c in long.columns]
    slim = long[keep]

    home_keys = df[["match_id", "home_team"]].merge(
        slim, left_on=["match_id", "home_team"], right_on=["match_id", "team"]
    )
    away_keys = df[["match_id", "away_team"]].merge(
        slim, left_on=["match_id", "away_team"], right_on=["match_id", "team"]
    )

    home_keys = home_keys.drop(columns=["home_team", "team"]).rename(
        columns=lambda c: c if c == "match_id" else f"home_{c}"
    )
    away_keys = away_keys.drop(columns=["away_team", "team"]).rename(
        columns=lambda c: c if c == "match_id" else f"away_{c}"
    )

    out = df.merge(home_keys, on="match_id").merge(away_keys, on="match_id")

    # Differences are usually stronger than the raw pair.
    for suffix in FEATURE_SUFFIXES:
        h, a = f"home_{suffix}", f"away_{suffix}"
        if h in out.columns and a in out.columns:
            out[f"diff_{suffix}"] = out[h] - out[a]

    out["target"] = out["result"].map({"H": 0, "D": 1, "A": 2})
    return out.sort_values("date").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    # The columns the model is allowed to see. Deliberately explicit.
    allowed = ["elo_home", "elo_away", "elo_diff",
               "is_promoted_home", "is_promoted_away", "h2h_home_ppg"]
    for suffix in FEATURE_SUFFIXES:
        for prefix in ("home_", "away_", "diff_"):
            col = f"{prefix}{suffix}"
            if col in df.columns:
                allowed.append(col)
    return [c for c in allowed if c in df.columns]


LEAKY_COLUMNS = {
    "home_goals", "away_goals", "result", "target",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "home_ht_goals", "away_ht_goals", "home_xg", "away_xg",
    "odds_home", "odds_draw", "odds_away",
}


def assert_no_leakage(cols: list[str]) -> None:
    # Fail loudly if a post-match column sneaks into the feature set.
    bad = sorted(set(cols) & LEAKY_COLUMNS)
    if bad:
        raise ValueError(f"Post-match columns in features: {bad}")


def implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["odds_home", "odds_draw", "odds_away"]
    inv = 1.0 / df[cols].astype(float)
    return inv.div(inv.sum(axis=1), axis=0).rename(
        columns={"odds_home": "p_home", "odds_draw": "p_draw", "odds_away": "p_away"}
    )


def season_elo_table(dataset: pd.DataFrame) -> pd.DataFrame:
    home = dataset[["season_start", "home_team", "elo_home"]].rename(
        columns={"home_team": "team", "elo_home": "elo"})
    away = dataset[["season_start", "away_team", "elo_away"]].rename(
        columns={"away_team": "team", "elo_away": "elo"})
    both = pd.concat([home, away], ignore_index=True)
    return both.groupby(["season_start", "team"], as_index=False)["elo"].mean()
