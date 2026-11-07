"""Loopback-only demo of the saved scientific model. No hosted service or external calls."""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .chemistry import featurize
from .data import TARGET, sha256
from .diagnostics import tanimoto
from .models import predict_model
from .pipeline import environment


class Workbench:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.web_root = Path(__file__).resolve().parent.parent / "web"
        self.bundle = joblib.load(self.run_dir / "model.joblib")
        if environment()["packages"] != self.bundle["manifest"]["environment"]["packages"]:
            raise ValueError("Dependency versions differ from saved model")
        expected = self.bundle["manifest"]["inputs"]["train"]["sha256"]
        if sha256(self.bundle["config"]["train"]) != expected:
            raise ValueError("Training source has changed since this model was saved")
        self.train = pd.read_csv(self.run_dir / "train_for_model.csv", dtype={"train_row_id": str})
        self.similarity_spec = {"radius": 2, "bits": 2048, "include_chirality": True}
        self.train_st, self.train_x = featurize(self.train.SMILES, self.similarity_spec)
        self.summary = self._summary()

    def _summary(self):
        def read(name):
            return json.loads((self.run_dir / name).read_text(encoding="utf-8"))
        selection, audit, diagnostics = read("selection.json"), read("audit.json"), read("diagnostics.json")
        table = pd.read_csv(self.run_dir / "leaderboard.csv").to_dict(orient="records")
        test = pd.read_csv(self.run_dir / "test_for_model.csv", dtype=str)
        return {"run_name": self.run_dir.name, "dataset_version": self.bundle["config"]["dataset_version"],
            "model_name": self.bundle["name"], "strategy": selection["strategy"], "promoted": selection["promoted"],
            "model_labels": selection.get("model_labels", {}), "comparison_candidate": selection.get("comparison_candidate", "tuned_ridge"),
            "promotion_checks": selection.get("promotion_checks", []),
            "audit": audit, "leaderboard": table, "diagnostics": diagnostics,
            "examples": [{"label": f"Молекула test · {test.iloc[i]['index']}", "smiles": test.iloc[i].SMILES} for i in (0, min(10, len(test)-1))],
            "input_features": "Только SMILES", "external_data_used": False,
            "validation_note": "Вложенная групповая валидация. Все разбиения используют один train; дополнительные разбиения проверяют устойчивость. Финального независимого теста нет."}

    def predict(self, smiles):
        if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 2000:
            raise ValueError("Введите SMILES длиной от 1 до 2000 символов.")
        structure, features = featurize([smiles], self.bundle["fingerprint"])
        if not structure.valid.iloc[0]:
            raise ValueError("Не удалось разобрать SMILES. Проверьте скобки, кольца и обозначения атомов или загрузите пример.")
        prediction, raw = predict_model(self.bundle, features, structure.valid)
        _, fingerprint = featurize([smiles], self.similarity_spec)
        similarities = tanimoto(fingerprint, self.train_x)[0]
        nearest = np.argsort(-similarities, kind="stable")[:3]
        canonical = structure.canonical_smiles.iloc[0]
        mol = Chem.MolFromSmiles(canonical)
        drawer = rdMolDraw2D.MolDraw2DSVG(600, 320)
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().encode("utf-8")
        return {"predicted_si": float(prediction[0]), "raw_prediction": float(raw[0]), "clipped_to_zero": bool(raw[0] < 0),
            "canonical_smiles": canonical, "molecule_image": "data:image/svg+xml;base64," + base64.b64encode(svg).decode(),
            "model_name": self.bundle["name"], "max_similarity": float(similarities[nearest[0]]),
            "training_matches": int((self.train_st.structure_id == structure.structure_id.iloc[0]).sum()),
            "neighbors": [{"row_id": self.train.iloc[i].train_row_id, "similarity": float(similarities[i]), "observed_si": float(self.train.iloc[i][TARGET])} for i in nearest],
            "similarity_note": "Сходство со структурами train, не вероятность правильного ответа."}


def make_server(run_dir, port=8765):
    app = Workbench(run_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; font-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value, status=200):
            self.send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.valid_host():
                self.json({"error": "Local access only"}, 403)
                return
            path = urlsplit(self.path).path
            if path == "/api/summary":
                self.json(app.summary)
                return
            routes = {
                "/": (app.web_root / "index.html", "text/html; charset=utf-8"),
                "/app.css": (app.web_root / "app.css", "text/css; charset=utf-8"),
                "/app.js": (app.web_root / "app.js", "text/javascript; charset=utf-8"),
                "/fonts/GolosText.ttf": (app.web_root / "fonts/GolosText.ttf", "font/ttf"),
                "/charts/oof_scatter.svg": (app.run_dir / "charts/oof_scatter.svg", "image/svg+xml"),
                "/charts/error_diagnostics.svg": (app.run_dir / "charts/error_diagnostics.svg", "image/svg+xml"),
            }
            for name in ("submission.csv", "report.md", "duplicate_review.csv", "error_review.csv", "leaderboard.csv", "inner_folds.csv", "benchmark.csv"):
                routes["/download/" + name] = (app.run_dir / name, "text/csv; charset=utf-8" if name.endswith(".csv") else "text/markdown; charset=utf-8")
            if path not in routes or not routes[path][0].is_file():
                self.json({"error": "Not found"}, 404)
                return
            file, mime = routes[path]
            self.send(file.read_bytes(), mime)

        def do_POST(self):
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if not self.valid_host() or self.headers.get("Origin", next(iter(allowed))) not in allowed:
                self.json({"error": "Local access only"}, 403)
                return
            if urlsplit(self.path).path != "/api/predict":
                self.json({"error": "Not found"}, 404)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.json({"error": "Expected JSON"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Запрос слишком большой или пустой.")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("Ожидается объект с полем smiles.")
                self.json(app.predict(value.get("smiles")))
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.json({"error": str(error)}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Local VACV SI workbench")
    parser.add_argument("--run", default="artifacts/kernel-v3")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.run, args.port)
    print(f"VACV SI: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
