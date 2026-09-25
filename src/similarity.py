import re

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR
from src.importance import normalise_player_name

TABLES = ("standard", "shooting", "misc", "playing_time", "keeper")
POSITION_GROUPS = ("GK", "DF", "MF", "FW")
OUTFIELD_GROUPS = ("DF", "MF", "FW")
KEYS = ["player_key", "born", "season_start"]

MIN_MINUTES = 900
PRIOR_NINETIES = 5.0
MIN_SIMILARITY = 0.15

COUNT_EXCLUDE = re.compile(r"(90|%|/)")

BASE_COLUMNS = {
    "player": ([r"player"], False),
    "team": ([r"team_canonical", r"team", r"squad"], False),
    "position": ([r"pos"], False),
    "age": ([r"age"], False),
    "born": ([r"born"], False),
    "minutes": ([r".*_Min"], True),
    "matches": ([r".*_MP"], True),
}

OUTFIELD_FEATURES = [
    ("goals", "standard", [r".*_Gls"], "count"),
    ("assists", "standard", [r".*_Ast"], "count"),
    ("non_penalty_goals", "standard", [r".*_G-PK"], "count"),
    ("shots", "shooting", [r".*_Sh"], "count"),
    ("shots_on_target", "shooting", [r".*_SoT"], "count"),
    ("shot_accuracy", "shooting", [r".*_SoT%"], "rate"),
    ("goals_per_shot", "shooting", [r".*_G/Sh"], "rate"),
    ("penalty_attempts", "shooting", [r".*_PKatt"], "count"),
    ("shot_distance", "shooting", [r".*_Dist"], "rate"),
    ("free_kicks", "shooting", [r".*_FK"], "count"),
    ("crosses", "misc", [r".*_Crs"], "count"),
    ("interceptions", "misc", [r".*_Int"], "count"),
    ("tackles_won", "misc", [r".*_TklW"], "count"),
    ("recoveries", "misc", [r".*_Recov"], "count"),
    ("fouls_committed", "misc", [r".*_Fls"], "count"),
    ("fouls_drawn", "misc", [r".*_Fld"], "count"),
    ("offsides", "misc", [r".*_Off"], "count"),
    ("aerials_won", "misc", [r"Aerial.*_Won"], "count"),
    ("aerial_win_pct", "misc", [r"Aerial.*_Won%"], "rate"),
    ("yellow_cards", "misc", [r".*_CrdY"], "count"),
    ("red_cards", "misc", [r".*_CrdR"], "count"),
    ("penalties_won", "misc", [r".*_PKwon"], "count"),
    ("penalties_conceded", "misc", [r".*_PKcon"], "count"),
]

KEEPER_FEATURES = [
    ("goals_against_90", "keeper", [r".*_GA90"], "rate"),
    ("save_pct", "keeper", [r"Performance_Save%", r".*Save%"], "rate"),
    ("clean_sheet_pct", "keeper", [r".*_CS%"], "rate"),
    ("saves", "keeper", [r".*_Saves"], "count"),
    ("shots_on_target_against", "keeper", [r".*_SoTA"], "count"),
    ("penalty_saves", "keeper", [r"Penalty Kicks_PKsv"], "count"),
    ("penalty_save_pct", "keeper", [r"Penalty Kicks_Save%"], "rate"),
]

COMPUTED_FEATURES = ["minutes_share", "minutes_per_match"]

ALL_FEATURES = OUTFIELD_FEATURES + KEEPER_FEATURES
KEEPER_NAMES = {f[0] for f in KEEPER_FEATURES}

RADAR = {
    "FW": ["goals_p90", "shots_p90", "shots_on_target_p90", "assists_p90",
           "fouls_drawn_p90", "offsides_p90", "aerials_won_p90", "crosses_p90",
           "minutes_share"],
    "MF": ["assists_p90", "crosses_p90", "interceptions_p90", "tackles_won_p90",
           "recoveries_p90", "fouls_drawn_p90", "shots_p90", "goals_p90",
           "minutes_share"],
    "DF": ["interceptions_p90", "tackles_won_p90", "recoveries_p90", "aerials_won_p90",
           "aerial_win_pct", "fouls_committed_p90", "crosses_p90", "yellow_cards_p90",
           "penalties_conceded_p90", "shots_p90", "minutes_share"],
    "GK": ["save_pct", "clean_sheet_pct", "goals_against_90", "saves_p90",
           "shots_on_target_against_p90", "penalty_save_pct", "minutes_share"],
}


def load_tables(folder=INTERIM_DIR):
    tables = {}
    for name in TABLES:
        path = folder / f"players_{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path, low_memory=False)
    return tables


def find_column(df, patterns, count=False):
    columns = [str(c) for c in df.columns]
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for col in columns:
            if rx.fullmatch(col.strip()) and not (count and COUNT_EXCLUDE.search(col)):
                return col
    return None


def _key_frame(df):
    player = find_column(df, [r"player"])
    team = find_column(df, [r"team", r"squad"])
    season = find_column(df, [r"season_start"])
    if None in (player, team, season):
        return None
    return pd.DataFrame({
        "_player": df[player].astype(str),
        "_team": df[team].astype(str),
        "season_start": pd.to_numeric(df[season], errors="coerce"),
    }, index=df.index)


def _parse_age(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def build_profiles(tables=None, prior_nineties=PRIOR_NINETIES, verbose=True):
    tables = tables if tables is not None else load_tables()
    if "standard" not in tables:
        raise FileNotFoundError(
            "players_standard.csv is missing from data/interim, run scripts/04_fetch_players.py")

    std = tables["standard"]
    base = _key_frame(std)
    if base is None:
        raise KeyError("players_standard.csv has no player, team or season_start column")

    for name, (patterns, is_count) in BASE_COLUMNS.items():
        col = find_column(std, patterns, count=is_count)
        if col is not None:
            base[name] = std[col]
    for needed in ("player", "minutes", "position"):
        if needed not in base:
            raise KeyError(f"players_standard.csv has no column for {needed!r}")

    base["minutes"] = pd.to_numeric(base["minutes"], errors="coerce").fillna(0)
    if "matches" in base:
        base["matches"] = pd.to_numeric(base["matches"], errors="coerce").fillna(0)
    if "team" not in base:
        base["team"] = base["_team"]

    merged = base.drop_duplicates(["_player", "_team", "season_start"])
    found, missing = [], []

    for table in TABLES:
        specs = [f for f in ALL_FEATURES if f[1] == table]
        if not specs:
            continue
        df = tables.get(table)
        keys = _key_frame(df) if df is not None else None
        if keys is None:
            missing += [s[0] for s in specs]
            continue

        values = {}
        for name, _, patterns, kind in specs:
            col = find_column(df, patterns, count=(kind == "count"))
            if col is None:
                missing.append(name)
                continue
            values[name] = pd.to_numeric(df[col], errors="coerce")
            found.append(name)

        if values:
            part = pd.concat([keys, pd.DataFrame(values, index=df.index)], axis=1)
            part = part.drop_duplicates(["_player", "_team", "season_start"])
            merged = merged.merge(part, on=["_player", "_team", "season_start"], how="left")

    merged["player_key"] = merged["player"].map(normalise_player_name)
    merged["born"] = (pd.to_numeric(merged["born"], errors="coerce") if "born" in merged
                      else np.nan)
    merged["born"] = merged["born"].fillna(-1).astype(int)
    merged["age"] = _parse_age(merged["age"]) if "age" in merged else np.nan

    count_cols = [n for n, _, _, k in ALL_FEATURES if k == "count" and n in merged]
    count_cols += ["minutes"] + (["matches"] if "matches" in merged else [])
    rate_cols = [n for n, _, _, k in ALL_FEATURES if k == "rate" and n in merged]

    merged = merged.sort_values("minutes", ascending=False)
    for r in rate_cols:
        has = merged[r].notna()
        merged[f"_{r}_w"] = merged[r].fillna(0) * merged["minutes"] * has
        merged[f"_{r}_m"] = merged["minutes"] * has

    agg = {c: "sum" for c in count_cols}
    agg.update({f"_{r}_w": "sum" for r in rate_cols})
    agg.update({f"_{r}_m": "sum" for r in rate_cols})
    for c in ("player", "team", "position", "age"):
        if c in merged:
            agg[c] = "first"

    prof = merged.groupby(KEYS, as_index=False).agg(agg)
    for r in rate_cols:
        prof[r] = prof[f"_{r}_w"] / prof[f"_{r}_m"].replace(0, np.nan)
    prof = prof.drop(columns=[c for c in prof.columns if c.startswith("_")])

    prof["pos_group"] = (prof["position"].astype(str).str.split(",").str[0]
                         .str.strip().str.upper().map({g: g for g in POSITION_GROUPS}))
    prof = prof.dropna(subset=["pos_group"]).copy()
    prof["nineties"] = prof["minutes"] / 90.0
    prof["minutes_share"] = (prof["minutes"] / (38 * 90)).clip(0, 1)
    if "matches" in prof.columns:
        prof["minutes_per_match"] = prof["minutes"] / prof["matches"].clip(lower=1)

    count_features = [n for n, _, _, k in ALL_FEATURES if k == "count" and n in prof]
    grp = prof.groupby("pos_group")
    group_nineties = grp["nineties"].transform("sum").replace(0, np.nan)
    new_cols = {}
    for f in count_features:
        prior = grp[f].transform("sum") / group_nineties
        new_cols[f"{f}_p90_raw"] = prof[f] / prof["nineties"].clip(lower=0.5)
        new_cols[f"{f}_p90"] = (prof[f] + prior_nineties * prior) / (prof["nineties"] + prior_nineties)
    prof = pd.concat([prof, pd.DataFrame(new_cols, index=prof.index)], axis=1)

    for r in rate_cols:
        prof[r] = prof[r].fillna(prof.groupby("pos_group")[r].transform("median"))

    prof = prof.sort_values(["season_start", "team", "minutes"],
                            ascending=[True, True, False]).reset_index(drop=True)

    if verbose:
        print(f"built {len(prof):,} player seasons from {len(tables)} FBref tables")
    return prof, found, missing


def feature_columns(profiles, group, raw=False):
    specs = KEEPER_FEATURES if group == "GK" else OUTFIELD_FEATURES
    cols = []
    for name, _, _, kind in specs:
        if kind == "count":
            col = f"{name}_p90_raw" if raw else f"{name}_p90"
        else:
            col = name
        if col in profiles.columns:
            cols.append(col)
    for col in COMPUTED_FEATURES:
        if col in profiles.columns:
            cols.append(col)
    return cols


def standardise(profiles, cols, min_minutes=MIN_MINUTES):
    z = pd.DataFrame(0.0, index=profiles.index, columns=cols)
    if not cols:
        return z
    for _, idx in profiles.groupby("pos_group").groups.items():
        block = profiles.loc[idx, cols].astype(float)
        ref = block[profiles.loc[idx, "minutes"] >= min_minutes]
        if len(ref) < 10:
            ref = block
        mu = ref.mean()
        sd = ref.std().replace(0, 1).fillna(1)
        z.loc[idx, cols] = ((block - mu) / sd).fillna(0).to_numpy()
    return z


def style_space(profiles, raw=False, min_minutes=MIN_MINUTES, columns=None):
    out_cols = columns if columns is not None else feature_columns(profiles, "FW", raw)
    gk_cols = feature_columns(profiles, "GK", raw)
    z_out = standardise(profiles, out_cols, min_minutes).add_prefix("o_")
    z_gk = standardise(profiles, gk_cols, min_minutes).add_prefix("k_")
    gk = profiles["pos_group"].eq("GK")
    z_out.loc[gk] = 0.0
    z_gk.loc[~gk] = 0.0
    return pd.concat([z_out, z_gk], axis=1)


def percentiles(profiles, cols, min_minutes=MIN_MINUTES):
    out = pd.DataFrame(np.nan, index=profiles.index, columns=cols)
    for _, idx in profiles.groupby("pos_group").groups.items():
        block = profiles.loc[idx]
        ref = block[block["minutes"] >= min_minutes]
        if ref.empty:
            continue
        for c in cols:
            if c not in block:
                continue
            sorted_ref = np.sort(ref[c].dropna().to_numpy(dtype=float))
            if len(sorted_ref) == 0:
                continue
            vals = block[c].to_numpy(dtype=float)
            out.loc[idx, c] = np.searchsorted(sorted_ref, vals, side="right") / len(sorted_ref) * 100
    return out


def label(col):
    return col.replace("_p90_raw", " per 90").replace("_p90", " per 90").replace("_", " ")


def _unit(matrix):
    m = np.asarray(matrix, dtype=float)
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return m / norm


def _identity(frame):
    return frame["player_key"].astype(str) + "|" + frame["born"].astype(str)


def self_consistency(profiles, space, season_pairs, min_minutes=MIN_MINUTES,
                     groups=OUTFIELD_GROUPS):
    ranks, pools = [], []
    ident = _identity(profiles)
    regular = profiles["minutes"] >= min_minutes

    for s0, s1 in season_pairs:
        for g in groups:
            in_group = profiles["pos_group"].eq(g) & regular
            a = profiles[in_group & profiles["season_start"].eq(s0)]
            b = profiles[in_group & profiles["season_start"].eq(s1)]
            if a.empty or b.empty:
                continue

            position = {k: i for i, k in enumerate(ident.loc[b.index])}
            queries = a[ident.loc[a.index].isin(position)]
            if queries.empty:
                continue

            sims = _unit(space.loc[queries.index]) @ _unit(space.loc[b.index]).T
            true_j = np.array([position[k] for k in ident.loc[queries.index]])
            true_sim = sims[np.arange(len(queries)), true_j]
            r = (sims > true_sim[:, None]).sum(axis=1) + 1
            ranks.extend(r.tolist())
            pools.extend([len(b)] * len(r))

    ranks, pools = np.array(ranks), np.array(pools)
    if len(ranks) == 0:
        return {"queries": 0}
    return {
        "queries": int(len(ranks)),
        "hit@1": float(np.mean(ranks <= 1)),
        "hit@5": float(np.mean(ranks <= 5)),
        "hit@10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "random_hit@10": float(np.mean(np.minimum(10, pools) / pools)),
    }


def resolve_player(profiles, name, season=None):
    key = normalise_player_name(name)
    hits = profiles[profiles["player_key"] == key]

    if hits.empty and key:
        surname = key.split()[-1]
        by_surname = profiles[profiles["player_key"].str.endswith(" " + surname)
                              | profiles["player_key"].eq(surname)]
        people = by_surname["player"].unique()
        if len(people) > 1:
            raise KeyError(f"{name!r} is ambiguous, did you mean one of {sorted(people)[:8]}")
        hits = by_surname

    if hits.empty:
        from rapidfuzz import process
        close = process.extract(name, profiles["player"].unique().tolist(), limit=5)
        raise KeyError(f"No player called {name!r}. Closest: {[c[0] for c in close]}")

    if season is not None:
        hits = hits[hits["season_start"] == season]
        if hits.empty:
            raise KeyError(f"{name!r} has no season starting {season}")

    return hits.sort_values(["season_start", "minutes"], ascending=False).index[0]


def find_replacements(profiles, space, player, season=None, n=10, candidate_season=None,
                      min_minutes=MIN_MINUTES, same_position=True, exclude_same_team=True,
                      max_age=None, max_value=None, min_similarity=MIN_SIMILARITY):
    query = resolve_player(profiles, player, season)
    row = profiles.loc[query]
    candidate_season = candidate_season or int(profiles["season_start"].max())

    pool = profiles[profiles["season_start"].eq(candidate_season)
                    & (profiles["minutes"] >= min_minutes)]
    pool = pool[_identity(pool) != _identity(profiles.loc[[query]]).iloc[0]]

    if same_position:
        pool = pool[pool["pos_group"] == row["pos_group"]]
    if exclude_same_team:
        pool = pool[pool["team"] != row["team"]]
    if max_age is not None:
        pool = pool[pool["age"] <= max_age]
    if max_value is not None:
        if "market_value_eur" not in pool:
            raise ValueError("No market values attached, run attach_market_value first")
        pool = pool[pool["market_value_eur"] <= max_value]

    columns = ["player", "team", "age", "position", "minutes"]
    if "market_value_eur" in profiles.columns:
        columns.append("market_value_eur")
    if pool.empty:
        return pd.DataFrame(columns=columns + ["similarity"])

    sims = _unit(space.loc[[query]]) @ _unit(space.loc[pool.index]).T
    out = pool.assign(similarity=sims[0]).nlargest(n, "similarity")
    out = out[out["similarity"] > min_similarity]
    return out[columns + ["similarity"]]


def compare_players(profiles, a, b, min_minutes=MIN_MINUTES):
    group = profiles.loc[a, "pos_group"]
    same_group = profiles[profiles["pos_group"] == group]
    cols = [c for c in feature_columns(profiles, group) if same_group[c].nunique() > 1]
    pct = percentiles(profiles, cols, min_minutes)
    name_a, name_b = profiles.loc[a, "player"], profiles.loc[b, "player"]
    if name_a == name_b:
        name_a, name_b = f"{name_a} (query)", f"{name_b} (match)"
    out = pd.DataFrame({
        "feature": [label(c) for c in cols],
        name_a: pct.loc[a, cols].to_numpy(dtype=float),
        name_b: pct.loc[b, cols].to_numpy(dtype=float),
    })
    out["gap"] = (out[name_a] - out[name_b]).abs()
    return out.sort_values("gap").round(0).reset_index(drop=True)


def attach_market_value(profiles, market):
    out = profiles.copy()
    m = market.dropna(subset=["market_value_eur"]).copy()
    m["player_key"] = m["player"].map(normalise_player_name)

    exact = m.groupby(["player_key", "season_start"])["market_value_eur"].max().to_dict()
    values = pd.Series([exact.get(k) for k in zip(out["player_key"], out["season_start"])],
                       index=out.index, dtype=float)

    if "club" in m.columns:
        m["surname"] = m["player_key"].str.split().str[-1]
        grouped = m.groupby(["surname", "club", "season_start"])["market_value_eur"]
        unique = grouped.agg(["max", "count"])
        by_club = unique[unique["count"] == 1]["max"].to_dict()
        missing = values.isna()
        surnames = out.loc[missing, "player_key"].str.split().str[-1]
        fills = [by_club.get(k) for k in zip(surnames, out.loc[missing, "team"],
                                             out.loc[missing, "season_start"])]
        values.loc[missing] = pd.Series(fills, index=surnames.index, dtype=float)

    out["market_value_eur"] = values
    return out


def save_profiles(profiles, space, path=None):
    path = path or (PROCESSED_DIR / "player_profiles.csv")
    vec = pd.DataFrame(space.to_numpy(dtype=float), index=profiles.index,
                       columns=[f"vec_{i}" for i in range(space.shape[1])])
    pd.concat([profiles, vec], axis=1).to_csv(path, index=False)
    return path


def load_profiles(path=None):
    path = path or (PROCESSED_DIR / "player_profiles.csv")
    df = pd.read_csv(path)
    vec_cols = [c for c in df.columns if c.startswith("vec_")]
    return df.drop(columns=vec_cols), df[vec_cols]
