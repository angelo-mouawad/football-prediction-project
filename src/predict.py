from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import MODELS_DIR

ALPHA = 0.4
CLASSES = ["home_win", "draw", "away_win"]

# Softmax temperature. Fitted on the validation seasons after training. A value below 1 sharpens an underconfident model, above 1 softens an overconfident one. 1.0 means no correction.
DEFAULT_TEMPERATURE = 1.0


class MatchNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, int] = (64, 32),
                 dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 3),
        )

    def forward(self, x):
        return self.net(x)


def save_bundle(model: MatchNet, scaler, feature_names: list[str],
                metrics: dict, temperature: float = DEFAULT_TEMPERATURE,
                name: str = "match_net") -> None:
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    np.savez(
        MODELS_DIR / f"{name}_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    meta = {
        "features": feature_names,
        "hidden": [64, 32],
        "metrics": metrics,
        "alpha": ALPHA,
        "temperature": float(temperature),
    }
    (MODELS_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))


def load_bundle(name: str = "match_net"):
    meta = json.loads((MODELS_DIR / f"{name}_meta.json").read_text())
    features = meta["features"]

    model = MatchNet(len(features), hidden=tuple(meta["hidden"]))
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt",
                                     map_location="cpu"))
    model.eval()

    stats = np.load(MODELS_DIR / f"{name}_scaler.npz")
    return model, stats["mean"], stats["scale"], features, meta


def network_probabilities(model: MatchNet, x: np.ndarray, mean, scale,
                          temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    # Scale, forward, temperature-adjusted softmax. Returns (n, 3).
    z = (x - mean) / scale
    with torch.no_grad():
        logits = model(torch.tensor(z, dtype=torch.float32))
        return torch.softmax(logits / float(temperature), dim=1).numpy()


def fit_temperature(logits: np.ndarray, y_true: np.ndarray,
                    bounds: tuple[float, float] = (0.3, 5.0)) -> float:
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss

    t_logits = torch.tensor(logits, dtype=torch.float32)

    def objective(t: float) -> float:
        p = torch.softmax(t_logits / float(t), dim=1).numpy()
        return log_loss(y_true, p, labels=[0, 1, 2])

    return float(minimize_scalar(objective, bounds=bounds, method="bounded").x)


def merge_branches(p_network: np.ndarray, news_home: float,
                   news_away: float, alpha: float = ALPHA) -> np.ndarray:
    p = np.clip(np.asarray(p_network, dtype=float), 1e-9, 1.0)
    delta = float(news_home) - float(news_away)
    adjust = np.array([alpha * delta, 0.0, -alpha * delta])

    shifted = np.log(p) + adjust
    shifted -= shifted.max()
    out = np.exp(shifted)
    return out / out.sum()


@dataclass
class Prediction:
    home_team: str
    away_team: str
    network: dict
    final: dict
    news_home: float
    news_away: float
    news_explanation: str
    verdict: str
    confidence: float
    articles: list = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.home_team} vs {self.away_team}\n"
            f"  network:  home {self.network['home_win']:.1%}  "
            f"draw {self.network['draw']:.1%}  away {self.network['away_win']:.1%}\n"
            f"  news:     home {self.news_home:+.2f}  away {self.news_away:+.2f}\n"
            f"  final:    home {self.final['home_win']:.1%}  "
            f"draw {self.final['draw']:.1%}  away {self.final['away_win']:.1%}\n"
            f"  verdict:  {self.verdict} ({self.confidence:.1%})"
        )


def latest_features(dataset: pd.DataFrame, home_team: str, away_team: str,
                    features: list[str]) -> np.ndarray:
    row = {}
    for team, prefix in ((home_team, "home"), (away_team, "away")):
        as_home = dataset[dataset["home_team"] == team]
        as_away = dataset[dataset["away_team"] == team]
        if as_home.empty and as_away.empty:
            raise ValueError(f"No history for {team!r}. Newly promoted?")

        last_home = as_home.iloc[-1] if len(as_home) else None
        last_away = as_away.iloc[-1] if len(as_away) else None

        if last_home is None:
            source, src_prefix = last_away, "away"
        elif last_away is None:
            source, src_prefix = last_home, "home"
        elif last_home["date"] >= last_away["date"]:
            source, src_prefix = last_home, "home"
        else:
            source, src_prefix = last_away, "away"

        row[f"elo_{prefix}"] = source[f"elo_{src_prefix}"]
        row[f"is_promoted_{prefix}"] = source[f"is_promoted_{src_prefix}"]

        for col in features:
            if not col.startswith(f"{prefix}_"):
                continue
            suffix = col[len(prefix) + 1:]
            src_col = f"{src_prefix}_{suffix}"
            if src_col in source.index:
                row[col] = source[src_col]

    row["elo_diff"] = row["elo_home"] - row["elo_away"]

    prior = dataset[
        (dataset["home_team"] == home_team) & (dataset["away_team"] == away_team)
    ]
    row["h2h_home_ppg"] = (
        float(prior["h2h_home_ppg"].iloc[-1]) if len(prior)
        and not pd.isna(prior["h2h_home_ppg"].iloc[-1]) else 1.5
    )

    for col in features:
        if col.startswith("diff_"):
            suffix = col[len("diff_"):]
            h, a = row.get(f"home_{suffix}"), row.get(f"away_{suffix}")
            if h is not None and a is not None:
                row[col] = h - a

    values = [row.get(col, np.nan) for col in features]
    arr = np.array(values, dtype=float).reshape(1, -1)
    return np.nan_to_num(arr, nan=0.0)


def predict_match(home_team: str, away_team: str, dataset: pd.DataFrame,
                  importance_table=None, use_news: bool = True,
                  bundle=None, news_result=None) -> Prediction:
    model, mean, scale, features, meta = bundle or load_bundle()
    temperature = meta.get("temperature", DEFAULT_TEMPERATURE)

    x = latest_features(dataset, home_team, away_team, features)
    p_net = network_probabilities(model, x, mean, scale, temperature)[0]

    news_home = news_away = 0.0
    explanation = "News branch disabled."
    articles = []

    if news_result is not None:
        result = news_result
    elif use_news and importance_table is not None:
        from src.news import analyse_fixture

        result = analyse_fixture(home_team, away_team, importance_table)
    else:
        result = None

    if result is not None:
        news_home, news_away = result.home_score, result.away_score
        explanation = result.explain()
        articles = [{"title": a["title"], "href": a["href"]}
                    for a in result.articles[:5]]

    p_final = merge_branches(p_net, news_home, news_away)
    best = int(np.argmax(p_final))

    verdict = {
        0: f"{home_team} win",
        1: "Draw",
        2: f"{away_team} win",
    }[best]

    return Prediction(
        home_team=home_team,
        away_team=away_team,
        network=dict(zip(CLASSES, p_net.round(4).tolist())),
        final=dict(zip(CLASSES, p_final.round(4).tolist())),
        news_home=round(news_home, 3),
        news_away=round(news_away, 3),
        news_explanation=explanation,
        verdict=verdict,
        confidence=float(p_final[best]),
        articles=articles,
    )
