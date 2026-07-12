#!/usr/bin/env python3
"""Build model-independent query complexity φ(q) for fit/calib/eval.

Stages (see WORKFLOW.md):
  A  row-local features
  B  frozen embeddings
  C  geometry fit on FIT only
  D  z-score (FIT) + C_* scalars
  E  write processed jsonl + artifacts

Usage (repo root):
  .venv/bin/python signals/query/build.py
  .venv/bin/python signals/query/build.py --limit 64   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import extract_row  # noqa: E402

ROLES = ("fit", "calib", "eval")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def flatten_numeric(rec: dict[str, Any]) -> dict[str, float]:
    """Flat path -> float for z-score / C_* inputs."""
    out: dict[str, float] = {}
    for block in ("structural", "linguistic_cues", "embedding_geometry"):
        for k, v in (rec.get(block) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                out[f"{block}.{k}"] = float(v)
    return out


# --- Stage B/C: embeddings + geometry ---------------------------------------


def encode_texts(texts: list[str], model_id: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return np.asarray(vecs, dtype=np.float64)


class GeometryModel:
    """PCA + centroid + kNN similarity + LOF; fit on FIT embeddings only."""

    def __init__(self) -> None:
        self.pca_components: list[list[float]] | None = None
        self.centroid: list[float] | None = None
        self.knn_k = 10
        self.lof_offset = 0.0
        self.lof_scale = 1.0
        self._nn = None
        self._lof = None
        self.fit_query_ids: list[str] = []

    def fit(self, matrix, query_ids: list[str], *, n_comp: int, knn_k: int) -> None:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

        x = np.asarray(matrix, dtype=np.float64)
        self.fit_query_ids = list(query_ids)
        self.knn_k = knn_k
        n_comp = min(n_comp, x.shape[0], x.shape[1])

        pca = PCA(n_components=n_comp)
        pca.fit(x)
        self.pca_components = pca.components_.tolist()
        self.centroid = x.mean(axis=0).tolist()

        k = min(knn_k, len(x))
        self._nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self._nn.fit(x)

        self._lof = LocalOutlierFactor(n_neighbors=k, novelty=True, metric="cosine")
        self._lof.fit(x)
        scores = -self._lof.score_samples(x)
        self.lof_offset = float(scores.mean())
        self.lof_scale = float(scores.std()) or 1.0

    def transform(self, vector) -> dict[str, float]:
        import numpy as np

        if self.pca_components is None or self.centroid is None:
            raise RuntimeError("GeometryModel not fit")
        x = np.asarray(vector, dtype=np.float64)
        comps = np.asarray(self.pca_components, dtype=np.float64)
        centroid = np.asarray(self.centroid, dtype=np.float64)
        pcs = comps @ (x - centroid)

        # cosine distance to centroid (embeddings are L2-normalized)
        c_norm = np.linalg.norm(centroid) or 1.0
        x_norm = np.linalg.norm(x) or 1.0
        centroid_distance = 1.0 - float(np.dot(x / x_norm, centroid / c_norm))

        dists, _ = self._nn.kneighbors(x.reshape(1, -1))
        mean_knn_similarity = float((1.0 - dists[0]).mean())

        lof_raw = -float(self._lof.score_samples(x.reshape(1, -1))[0])
        lof_score = (lof_raw - self.lof_offset) / self.lof_scale

        out = {
            "centroid_distance": centroid_distance,
            "mean_knn_similarity": mean_knn_similarity,
            "lof_score": lof_score,
        }
        for i, v in enumerate(pcs.tolist(), start=1):
            out[f"pc{i}"] = float(v)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "pca_components": self.pca_components,
            "centroid": self.centroid,
            "knn_k": self.knn_k,
            "lof_offset": self.lof_offset,
            "lof_scale": self.lof_scale,
            "fit_query_ids": self.fit_query_ids,
            "n_fit": len(self.fit_query_ids),
        }


# --- Stage D: z-score + complexity scalars ----------------------------------


class ZScore:
    def __init__(self) -> None:
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, flats: list[dict[str, float]]) -> None:
        buckets: dict[str, list[float]] = defaultdict(list)
        for flat in flats:
            for k, v in flat.items():
                buckets[k].append(v)
        for k, vals in buckets.items():
            self.mean[k] = sum(vals) / len(vals)
            if len(vals) < 2:
                self.std[k] = 1.0
            else:
                m = self.mean[k]
                self.std[k] = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0

    def transform_flat(self, flat: dict[str, float]) -> dict[str, float]:
        return {
            k: (v - self.mean.get(k, 0.0)) / (self.std.get(k, 1.0) or 1.0)
            for k, v in flat.items()
            if k in self.mean
        }

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}


def compose_complexity(
    flat_z: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    c_length = flat_z.get("structural.prompt_token_len", 0.0)
    # denser text: higher MATTR, lower compression_ratio → higher complexity
    c_density = flat_z.get("structural.mattr", 0.0) - flat_z.get("structural.compression_ratio", 0.0)
    c_atypical = 0.5 * flat_z.get("embedding_geometry.centroid_distance", 0.0) + 0.5 * flat_z.get(
        "embedding_geometry.lof_score", 0.0
    )
    ling_keys = [
        "linguistic_cues.n_requirements",
        "linguistic_cues.reasoning_depth_score",
        "linguistic_cues.multihop_score",
        "linguistic_cues.domain_breadth",
        "linguistic_cues.n_sentences",
    ]
    ling_vals = [flat_z[k] for k in ling_keys if k in flat_z]
    c_linguistic = sum(ling_vals) / len(ling_vals) if ling_vals else 0.0

    parts = {
        "C_length": c_length,
        "C_density": c_density,
        "C_atypical": c_atypical,
        "C_linguistic": c_linguistic,
    }
    wsum = sum(weights.get(k, 0.0) for k in parts) or 1.0
    parts["C_query"] = sum(parts[k] * weights.get(k, 0.0) for k in parts) / wsum
    return parts


def assert_no_eval_in_geometry(geo: GeometryModel, eval_ids: set[str]) -> None:
    overlap = set(geo.fit_query_ids) & eval_ids
    if overlap:
        raise SystemExit(f"LEAKAGE: geometry fit contains eval ids n={len(overlap)}")


# --- pipeline ---------------------------------------------------------------


def load_corpus(corpus_dir: Path, limit: int | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        path = corpus_dir / f"queries_{role}.jsonl"
        rows = read_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        out[role] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build model-independent query complexity")
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifacts", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per role (smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus_dir = args.corpus or (ROOT / cfg["corpus_dir"])
    out_dir = args.out or (ROOT / cfg["output_dir"])
    art_dir = args.artifacts or (ROOT / cfg["artifact_dir"])
    emb_dir = ROOT / cfg.get("embedding_dir", "signals/query/embeddings")

    print(f"loading corpus from {corpus_dir}", flush=True)
    by_role = load_corpus(corpus_dir, args.limit)
    eval_ids = {r["query_id"] for r in by_role["eval"]}
    fit_ids = {r["query_id"] for r in by_role["fit"]}

    # Stage A
    print("stage A: structural + linguistic + task_form", flush=True)
    records: dict[str, list[dict[str, Any]]] = {}
    id_to_prompt: dict[str, str] = {}
    for role, rows in by_role.items():
        recs = []
        for row in rows:
            rec = extract_row(row, cfg)
            recs.append(rec)
            id_to_prompt[row["query_id"]] = row["prompt"]
        records[role] = recs

    # Stage B — embed all prompts once
    print("stage B: embeddings", flush=True)
    all_ids = []
    all_texts = []
    for role in ROLES:
        for rec in records[role]:
            all_ids.append(rec["query_id"])
            all_texts.append(id_to_prompt[rec["query_id"]])
    # unique preserve order
    seen = set()
    uniq_ids, uniq_texts = [], []
    for i, t in zip(all_ids, all_texts):
        if i in seen:
            continue
        seen.add(i)
        uniq_ids.append(i)
        uniq_texts.append(t)

    vectors = encode_texts(uniq_texts, cfg["embedder_id"])
    id_to_vec = {qid: vectors[i] for i, qid in enumerate(uniq_ids)}
    emb_dir.mkdir(parents=True, exist_ok=True)
    # lightweight cache index
    write_json(
        emb_dir / "index.json",
        {"embedder_id": cfg["embedder_id"], "n": len(uniq_ids), "dim": int(vectors.shape[1])},
    )

    # Stage C — geometry on FIT only
    print("stage C: geometry fit on FIT", flush=True)
    fit_order = [r["query_id"] for r in records["fit"]]
    fit_matrix = [id_to_vec[qid] for qid in fit_order]
    geo = GeometryModel()
    geo.fit(
        fit_matrix,
        fit_order,
        n_comp=int(cfg.get("pca_components", 3)),
        knn_k=int(cfg.get("knn_k", 10)),
    )
    assert_no_eval_in_geometry(geo, eval_ids)
    if set(geo.fit_query_ids) - fit_ids:
        raise SystemExit("geometry fit ids not subset of fit")

    for role in ROLES:
        for rec in records[role]:
            rec["embedding_geometry"] = geo.transform(id_to_vec[rec["query_id"]])

    # Stage D — z-score on FIT flats, then C_*
    print("stage D: z-score + C_query", flush=True)
    fit_flats = [flatten_numeric(r) for r in records["fit"]]
    zmodel = ZScore()
    zmodel.fit(fit_flats)
    weights = cfg.get("complexity_weights") or {}

    for role in ROLES:
        for rec in records[role]:
            flat = flatten_numeric(rec)
            flat_z = zmodel.transform_flat(flat)
            rec["complexity"] = compose_complexity(flat_z, weights)

    # Stage E
    print("stage E: write outputs", flush=True)
    for role in ROLES:
        write_jsonl(out_dir / f"{role}.jsonl", records[role])

    write_json(art_dir / "geometry.json", geo.to_dict())
    write_json(art_dir / "zscore.json", zmodel.to_dict())
    write_json(art_dir / "complexity_weights.json", weights)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "embedder_id": cfg["embedder_id"],
        "counts": {role: len(records[role]) for role in ROLES},
        "geometry_n_fit": len(geo.fit_query_ids),
        "eval_overlap_geometry": 0,
        "config_sha16": hashlib.sha256(args.config.read_bytes()).hexdigest()[:16],
        "limit": args.limit,
    }
    write_json(art_dir / "manifest.json", manifest)
    print("done", json.dumps(manifest["counts"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
