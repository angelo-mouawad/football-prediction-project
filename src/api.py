from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT

WEB_DIR = PROJECT_ROOT / "web"

MATCHES = PROCESSED_DIR / "matches_features.csv"
VALUES = PROCESSED_DIR / "player_market_values.csv"
PROFILES = PROCESSED_DIR / "player_profiles.csv"
FIXTURES = PROCESSED_DIR / "fixtures_upcoming.csv"
VALUE_META = MODELS_DIR / "value_meta.json"
VALUE_MODEL = MODELS_DIR / "value_xgb.json"
MATCH_META = MODELS_DIR / "match_net_meta.json"

app = FastAPI(title="Football Intelligence")


@lru_cache(maxsize=1)
def matches_frame():
    if not MATCHES.exists():
        raise HTTPException(503, "Run notebook 1 first, matches_features.csv is missing")
    return pd.read_csv(MATCHES, parse_dates=["date"], low_memory=False)


@lru_cache(maxsize=1)
def match_bundle():
    if not MATCH_META.exists():
        raise HTTPException(503, "Run notebook 1 first, no trained match model")
    try:
        from src.predict import load_bundle
    except ImportError as exc:
        raise HTTPException(503, f"Missing dependency for the match model: {exc}")

    return load_bundle()


@lru_cache(maxsize=1)
def importance_table():
    for name in ("player_importance_value.csv", "player_importance.csv"):
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_csv(path)
    return None


@lru_cache(maxsize=1)
def values_frame():
    if not VALUES.exists():
        raise HTTPException(503, "Run notebook 2 first, player_market_values.csv is missing")
    df = pd.read_csv(VALUES, low_memory=False)
    from src.importance import normalise_player_name

    df["player_key"] = df["player"].map(normalise_player_name)
    return df


@lru_cache(maxsize=1)
def value_model():
    if not VALUE_MODEL.exists() or not VALUE_META.exists():
        raise HTTPException(503, "Run notebook 2 first, no trained value model")
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(VALUE_MODEL)
    meta = json.loads(VALUE_META.read_text())
    index = pd.Series({int(k): float(v) for k, v in meta.get("market_index", {}).items()})
    return model, meta, index


@lru_cache(maxsize=1)
def profiles_frame():
    if not PROFILES.exists():
        raise HTTPException(503, "Run notebook 3 first, player_profiles.csv is missing")
    from src import similarity as S

    return S.load_profiles()


def clean(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return value


def rows(frame: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


@app.get("/api/status")
def status():
    available = {
        "match": MATCHES.exists() and MATCH_META.exists(),
        "value": VALUES.exists() and VALUE_MODEL.exists(),
        "similar": PROFILES.exists(),
        "fixtures": FIXTURES.exists(),
        "news": importance_table() is not None,
    }
    meta = {}
    if MATCH_META.exists():
        meta = json.loads(MATCH_META.read_text()).get("metrics", {})
    return {"available": available, "match_metrics": meta}


@app.get("/api/teams")
def teams():
    df = matches_frame()
    recent = df[df["season_start"] >= df["season_start"].max() - 1]
    current = sorted(set(recent["home_team"]) | set(recent["away_team"]))
    everything = sorted(set(df["home_team"]) | set(df["away_team"]))
    return {"current": current, "all": everything}


@app.get("/api/fixtures")
def fixtures(limit: int = 12):
    if not FIXTURES.exists():
        return {"fixtures": []}
    df = pd.read_csv(FIXTURES, parse_dates=["date"]).head(limit)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
    return {"fixtures": rows(df)}


@app.get("/api/predict")
def predict(home: str, away: str, news: bool = True):
    if home == away:
        raise HTTPException(400, "Pick two different teams")

    bundle = match_bundle()
    from src.predict import predict_match

    table = importance_table()

    news_result, facts = None, []
    if news and table is not None:
        from src.news import analyse_fixture

        news_result = analyse_fixture(home, away, table)
        facts = [
            {"player": f.player, "side": f.side, "status": f.status,
             "importance": round(f.importance, 3), "impact": round(f.impact, 3)}
            for f in news_result.facts
        ]

    try:
        result = predict_match(home, away, matches_frame(), importance_table=table,
                               use_news=False, bundle=bundle, news_result=news_result)
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    payload = result.to_dict()
    payload["facts"] = sorted(facts, key=lambda f: -abs(f["impact"]))
    return payload


@app.get("/api/players")
def players(q: str = Query(min_length=1), limit: int = 8, source: str = "value"):
    if source == "profiles":
        frame, _ = profiles_frame()
        season = frame["season_start"].max()
        frame = frame[frame["season_start"] == season]
        label_cols = ["player", "team", "pos_group", "age"]
    else:
        frame = values_frame()
        season = frame["season_start"].max()
        frame = frame[frame["season_start"] == season]
        frame = frame.rename(columns={"club": "team"})
        label_cols = ["player", "team", "position", "age"]

    needle = q.strip().lower()
    hit = frame[frame["player"].str.lower().str.contains(needle, na=False)]
    if hit.empty:
        from rapidfuzz import process

        names = frame["player"].dropna().unique().tolist()
        close = [c[0] for c in process.extract(q, names, limit=limit)]
        hit = frame[frame["player"].isin(close)]

    hit = hit.drop_duplicates("player").head(limit)
    return {"players": rows(hit[[c for c in label_cols if c in hit.columns]])}


@app.get("/api/value")
def value(player: str):
    from src import market as M
    from src.importance import normalise_player_name

    df = values_frame()
    model, meta, index = value_model()

    key = normalise_player_name(player)
    hit = df[df["player_key"] == key]
    if hit.empty:
        hit = df[df["player"].str.lower() == player.strip().lower()]
    if hit.empty:
        raise HTTPException(404, f"No player called {player}")

    row = hit.sort_values("season_start").iloc[-1]
    season = int(row["season_start"])

    features = meta["features"]
    X, _ = M.feature_matrix(df, include_prev_value=False)
    X = X.fillna(X.median(numeric_only=True))
    X = X.reindex(columns=features, fill_value=0.0)
    vector = X.loc[[row.name]].fillna(0.0).to_numpy(dtype=np.float32)

    relative = float(model.predict(vector)[0])
    market_level = float(M.index_for([season], index)[0])
    predicted = float(np.exp(relative + market_level))
    actual = float(row["market_value_eur"])

    return {
        "player": row["player"],
        "club": row.get("club"),
        "season": season,
        "age": clean(row.get("age")),
        "position": row.get("pos_group") or row.get("position"),
        "minutes": clean(row.get("minutes")),
        "goals": clean(row.get("goals")),
        "assists": clean(row.get("assists")),
        "predicted": predicted,
        "actual": actual,
        "gap": (predicted - actual) / actual if actual else None,
        "market_level": float(np.exp(market_level)),
    }


@app.get("/api/similar")
def similar(player: str, n: int = 8, max_age: float | None = None,
            max_value: float | None = None):
    from src import similarity as S

    frame, space = profiles_frame()
    try:
        query = S.resolve_player(frame, player)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    if "market_value_eur" not in frame.columns and VALUES.exists():
        frame = S.attach_market_value(frame, pd.read_csv(VALUES, low_memory=False))

    try:
        found = S.find_replacements(frame, space, player, n=n, max_age=max_age,
                                    max_value=max_value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    source = frame.loc[query]
    payload = {
        "query": {
            "player": source["player"], "team": source["team"],
            "position": source["pos_group"], "age": clean(source.get("age")),
            "minutes": clean(source["minutes"]), "season": int(source["season_start"]),
            "value": clean(source.get("market_value_eur")),
        },
        "matches": rows(found.reset_index().rename(columns={"index": "row_id"})),
        "radar": [],
        "comparison": [],
    }

    if len(found):
        top = found.index[0]
        radar_cols = [c for c in S.RADAR.get(source["pos_group"], []) if c in frame.columns]
        if radar_cols:
            pct = S.percentiles(frame, radar_cols)
            payload["radar"] = [
                {"feature": S.label(c),
                 "query": clean(pct.loc[query, c]) or 0,
                 "match": clean(pct.loc[top, c]) or 0}
                for c in radar_cols
            ]
        comparison = S.compare_players(frame, query, top)
        comparison.columns = ["feature", "query", "match", "gap"]
        payload["comparison"] = rows(comparison)
        payload["top_match"] = frame.loc[top, "player"]

    return payload


@app.exception_handler(HTTPException)
def http_error(request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
