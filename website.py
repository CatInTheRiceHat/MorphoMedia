"""
Lightweight web app for Healthy Feed Algorithm using only stdlib + existing project deps.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from algorithm import (
    WEIGHTS,
    add_engagement,
    build_prototype_feed,
    get_mode_settings,
    rank_baseline,
    validate_and_clean,
)
from metrics import diversity_at_k, max_streak, prosocial_ratio


# Render expects web services to bind 0.0.0.0 on the provided PORT.
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))
ROOT = Path(__file__).parent
WEB_DIR = ROOT / "website"
DEFAULT_DATASET = ROOT / "datasets" / "shorts_dataset_tagged.csv"
EMBED_CACHE: dict[str, bool] = {}


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def metrics_for_feed(feed: pd.DataFrame) -> dict:
    return {
        "diversity_at_10": int(diversity_at_k(feed, k=10, topic_col="topic")),
        "max_topic_streak": int(max_streak(feed, "topic")),
        "max_creator_streak": int(max_streak(feed, "channel")),
        "prosocial_ratio": float(prosocial_ratio(feed, prosocial_col="prosocial")),
    }


def ensure_algorithm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing columns so fetched YouTube data can run through the algorithm.
    """
    out = df.copy()
    if "topic" not in out.columns:
        out["topic"] = "unlabeled"
    if "prosocial" not in out.columns:
        out["prosocial"] = 0
    if "risk" not in out.columns:
        out["risk"] = 0
    return out


def run_model(df: pd.DataFrame, preset: str, night_mode: bool, recent_window: int) -> dict:
    df = ensure_algorithm_columns(df)
    df = validate_and_clean(df)
    df, _ = add_engagement(df)

    weights, k = get_mode_settings(
        preset=preset, night_mode=night_mode, k_default=100)
    improved = build_prototype_feed(
        df, weights=weights, k=k, recent_window=recent_window
    ).reset_index(drop=True)
    baseline = rank_baseline(df, k=k).reset_index(drop=True)

    cols = [
        c
        for c in [
            "video_id",
            "title",
            "topic",
            "channel",
            "prosocial",
            "risk",
            "engagement",
            "diversity",
            "score",
        ]
        if c in improved.columns
    ]

    return {
        "preset": preset,
        "night_mode": night_mode,
        "k": k,
        "weights": weights,
        "improved_metrics": metrics_for_feed(improved),
        "baseline_metrics": metrics_for_feed(baseline),
        "improved_feed": improved[cols].head(min(k, 50)).to_dict(orient="records"),
        "baseline_feed": baseline[[c for c in cols if c in baseline.columns]]
        .head(min(k, 50))
        .to_dict(orient="records"),
        "improved_top10": improved[cols].head(10).to_dict(orient="records"),
        "baseline_top10": baseline[[c for c in cols if c in baseline.columns]]
        .head(10)
        .to_dict(orient="records"),
    }


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            return self._serve_index()
        if parsed.path == "/api/presets":
            return json_response(self, 200, {"presets": list(WEIGHTS.keys())})
        return json_response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/run/local":
            return self._run_local()
        if parsed.path == "/api/check/embed":
            return self._check_embed()
        return json_response(self, 404, {"error": "Not found"})

    def log_message(self, fmt: str, *args) -> None:
        # Keep console output clean.
        return

    def _read_json(self) -> dict:
        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _serve_index(self) -> None:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            return json_response(self, 500, {"error": "Missing website/index.html"})
        html = index_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _run_local(self) -> None:
        payload = self._read_json()
        preset = str(payload.get("preset", "entertainment"))
        night_mode = bool(payload.get("night_mode", False))
        recent_window = int(payload.get("recent_window", 10))
        dataset_path = Path(payload.get("dataset_path") or DEFAULT_DATASET)
        if not dataset_path.exists():
            return json_response(
                self,
                400,
                {"error": f"Dataset not found: {dataset_path}"},
            )
        try:
            df = pd.read_csv(dataset_path)
            result = run_model(
                df, preset=preset, night_mode=night_mode, recent_window=recent_window)
            result["source"] = str(dataset_path)
            return json_response(self, 200, result)
        except Exception as exc:
            return json_response(self, 400, {"error": str(exc)})

    def _check_embed(self) -> None:
        payload = self._read_json()
        raw_ids = payload.get("video_ids", [])
        if not isinstance(raw_ids, list):
            return json_response(self, 400, {"error": "video_ids must be a list"})

        # Keep ids short and safe.
        video_ids = [str(v).strip()[:32] for v in raw_ids if str(v).strip()]
        status: dict[str, bool] = {}

        for vid in video_ids:
            if vid in EMBED_CACHE:
                status[vid] = EMBED_CACHE[vid]
                continue

            url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json"
            ok = False
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=4) as resp:
                    ok = 200 <= int(getattr(resp, "status", 0)) < 300
            except Exception:
                ok = False

            EMBED_CACHE[vid] = ok
            status[vid] = ok

        return json_response(self, 200, {"status": status})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Healthy Feed web app running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
