"""Experimental: Semantic data layer for the Hybrid JEPA Semantic LLM.

Provides a 9-dimensional semantic vector space grounded in real conversational
vocabulary (8 vocab-grounded dimensions + 1 taxonomy hash dimension), locality-
sensitive hashing, and a precomputed similarity matrix for O(1) semantic
nearest-neighbor lookup.  All APIs here are research prototypes and may change
or be removed based on findings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data" / "semantic"


class SemanticDim(NamedTuple):
    index: int
    name: str
    meaning: str


SEMANTIC_DIMS: list[SemanticDim] = [
    SemanticDim(0, "entity_type", "who or what (person, object, abstract)"),
    SemanticDim(1, "action", "what happens (verb / activity)"),
    SemanticDim(2, "property", "how it is (size, speed, quality)"),
    SemanticDim(3, "emotion_sentiment", "how it feels (emotion, valence)"),
    SemanticDim(4, "domain", "what field (science, art, business)"),
    SemanticDim(5, "temporal", "when (time of day, season, frequency)"),
    SemanticDim(6, "causality", "why (cause, effect, enable, prevent)"),
    SemanticDim(7, "social_register", "how to speak (formal, casual, technical)"),
    SemanticDim(8, "taxonomy_hash", "signature hash (entity+action+domain trigram)"),
]

SEMANTIC_DIM_NAMES: list[str] = [d.name for d in SEMANTIC_DIMS]
VOCAB_DIM_NAMES: list[str] = SEMANTIC_DIM_NAMES[:8]
NUM_SEMANTIC_DIMS = 9
NUM_VOCAB_DIMS = 8


def signature_to_float(sig: str, n_buckets: int = 4096) -> float:
    """Deterministically hash a semantic_signature string to a float in [0, 1].

    Uses a stable hash (MD5 truncated) so the mapping is reproducible across
    Python versions and sessions, unlike the built-in ``hash()`` which is
    randomized by default.
    """
    h = int(hashlib.md5(sig.encode("utf-8")).hexdigest()[:8], 16)
    return (h % n_buckets) / n_buckets


class ConversationalVocabulary:
    """Loads the 320-term vocabulary (8 dims x 40 terms) and encodes rows."""

    def __init__(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else _DATA_DIR / "vocab_8d.json"
        with open(path) as f:
            data = json.load(f)
        self.raw: dict[str, Any] = data
        self._vocab: dict[str, list[str]] = data["vocabulary"]
        self._term_index: dict[str, dict[str, int]] = {}
        for dim_name, terms in self._vocab.items():
            self._term_index[dim_name] = {t.lower(): i for i, t in enumerate(terms)}

    @property
    def dim_names(self) -> list[str]:
        return list(self._vocab.keys())

    def terms(self, dim_name: str) -> list[str]:
        return list(self._vocab[dim_name])

    def term_to_index(self, dim_name: str, term: str) -> int:
        """Return the 0-based index of *term* in *dim_name*, or -1 if unknown."""
        idx_map = self._term_index.get(dim_name)
        if idx_map is None:
            return -1
        return idx_map.get(term.lower(), -1)

    def encode_row(
        self,
        row: dict[str, str],
        n_sig_buckets: int = 4096,
    ) -> np.ndarray:
        """Encode a training CSV row dict to a 9D float32 vector.

        Dims 0-7: term index normalised to [-1, 1].  Unknown terms map to 0.0.
        Dim 8: ``signature_to_float(row["semantic_signature"])``.
        """
        vec = np.zeros(NUM_SEMANTIC_DIMS, dtype=np.float32)
        for i, dim_name in enumerate(VOCAB_DIM_NAMES):
            term = row.get(dim_name, "")
            idx = self.term_to_index(dim_name, term)
            n_terms = len(self._vocab.get(dim_name, []))
            if idx >= 0 and n_terms > 1:
                vec[i] = 2.0 * idx / (n_terms - 1) - 1.0
        sig = row.get("semantic_signature", "")
        if sig:
            vec[8] = signature_to_float(sig, n_sig_buckets)
        return vec

    def decode_vector(self, vec: np.ndarray) -> dict[str, str | float]:
        """Map a 9D vector back to nearest vocabulary terms + signature bucket."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        result: dict[str, str | float] = {}
        for i, dim_name in enumerate(VOCAB_DIM_NAMES):
            terms = self._vocab.get(dim_name, [])
            if not terms:
                result[dim_name] = ""
                continue
            idx = int(round((vec[i] + 1.0) / 2.0 * (len(terms) - 1)))
            idx = max(0, min(idx, len(terms) - 1))
            result[dim_name] = terms[idx]
        if len(vec) > 8:
            result["taxonomy_hash"] = float(vec[8])
        return result


def load_training_data(
    path: str | Path | None = None,
    vocab: ConversationalVocabulary | None = None,
    n_sig_buckets: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the training CSV and return ``(ids, vectors)`` for SemanticMatrix.

    Returns:
        ids:  (N,)  int64 row ids
        vecs: (N, 9) float32 semantic vectors
    """
    path = Path(path) if path else _DATA_DIR / "training_100k_8d.csv"
    if vocab is None:
        vocab = ConversationalVocabulary()
    ids_list: list[int] = []
    vecs_list: list[np.ndarray] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids_list.append(int(row["id"]))
            vecs_list.append(vocab.encode_row(row, n_sig_buckets=n_sig_buckets))
    ids = np.array(ids_list, dtype=np.int64)
    vecs = np.stack(vecs_list).astype(np.float32)
    return ids, vecs


class SemanticMatrix:
    """In-memory semantic similarity matrix.

    Expected npz layout::

        ids   -- (N,)  int64 token ids
        vecs  -- (N, 9) float32 L2-normalised semantic vectors
    """

    def __init__(self, path: str | Path) -> None:
        data = np.load(str(path), allow_pickle=True)
        self.ids: np.ndarray = data["ids"]
        self.vectors: np.ndarray = data["vecs"].astype(np.float32)
        if self.vectors.shape[1] != NUM_SEMANTIC_DIMS:
            raise ValueError(
                f"Expected {NUM_SEMANTIC_DIMS}-D vectors, got {self.vectors.shape[1]}-D"
            )
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.vectors = self.vectors / norms

    def neighbors(self, vec: np.ndarray, k: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) for the *k* nearest neighbours by cosine."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if vec.shape[0] != NUM_SEMANTIC_DIMS:
            raise ValueError(f"Expected {NUM_SEMANTIC_DIMS}-D query, got {vec.shape[0]}-D")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores = self.vectors @ vec
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k)[:k]
        order = np.argsort(-scores[idx])
        idx = idx[order]
        return self.ids[idx], scores[idx]

    def build_hasher(
        self,
        tables: int = 8,
        planes: int = 12,
        seed: int = 42,
    ) -> "SemanticHasher":
        hasher = SemanticHasher(dim=NUM_SEMANTIC_DIMS, tables=tables, planes=planes, seed=seed)
        hasher.index(self.ids, self.vectors)
        return hasher


class SemanticHasher:
    """Multi-table random-hyperplane LSH for 9-D semantic vectors."""

    def __init__(
        self,
        dim: int = NUM_SEMANTIC_DIMS,
        tables: int = 8,
        planes: int = 12,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.n_tables = tables
        self.n_planes = planes
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(tables, planes, dim).astype(np.float32)
        self._buckets: list[dict[bytes, list[int]]] = [
            defaultdict(list) for _ in range(tables)
        ]
        self._ids: np.ndarray | None = None
        self._vecs: np.ndarray | None = None

    def hash(self, vec: np.ndarray) -> tuple[bytes, ...]:
        """Hash a single vector into one bucket key per table."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        bits = (self.proj @ vec) > 0
        return tuple(bits[t].tobytes() for t in range(self.n_tables))

    def index(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        """Build the hash index from a matrix of (N, dim) vectors."""
        self._ids = np.asarray(ids)
        self._vecs = np.asarray(vecs, dtype=np.float32)
        for i in range(len(ids)):
            keys = self.hash(self._vecs[i])
            for t, key in enumerate(keys):
                self._buckets[t][key].append(i)

    def query(self, vec: np.ndarray, k: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve *k* approximate nearest neighbours with exact reranking."""
        if self._ids is None or self._vecs is None:
            raise RuntimeError("Hasher has no indexed data. Call index() first.")
        keys = self.hash(vec)
        candidates: set[int] = set()
        for t, key in enumerate(keys):
            candidates.update(self._buckets[t].get(key, []))
        if not candidates:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        cand_idx = np.array(sorted(candidates), dtype=np.int64)
        cand_vecs = self._vecs[cand_idx]
        qvec = np.asarray(vec, dtype=np.float32).ravel()
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm
        scores = cand_vecs @ qvec
        k = min(k, len(scores))
        if k == len(scores):
            top = np.argsort(-scores)[:k]
            return self._ids[cand_idx[top]], scores[top]
        top = np.argpartition(-scores, k)[:k]
        order = np.argsort(-scores[top])
        top = top[order]
        return self._ids[cand_idx[top]], scores[top]

    def get_projection_planes(self) -> np.ndarray:
        """Return (tables, planes, dim) projection matrix for torch buffer use."""
        return self.proj.copy()


def generate_synthetic_semantic_data(
    n: int = 100_000,
    dim: int = NUM_SEMANTIC_DIMS,
    seed: int = 0,
    out_dir: str | Path = "neuralfn/data/semantic",
) -> Path:
    """Generate synthetic data using real vocabulary when available."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    try:
        vocab = ConversationalVocabulary()
        ids = np.arange(n, dtype=np.int64)
        vecs = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            for d, dname in enumerate(VOCAB_DIM_NAMES):
                terms = vocab.terms(dname)
                if terms:
                    idx = rng.randint(0, len(terms))
                    vecs[i, d] = 2.0 * idx / max(len(terms) - 1, 1) - 1.0
            parts = []
            for short_dim in ("entity_type", "action", "domain"):
                terms = vocab.terms(short_dim)
                if terms:
                    parts.append(terms[rng.randint(0, len(terms))][:3].lower())
            sig = "_".join(parts) if parts else str(i)
            vecs[i, 8] = signature_to_float(sig)
    except (FileNotFoundError, json.JSONDecodeError):
        ids = np.arange(n, dtype=np.int64)
        vecs = rng.randn(n, dim).astype(np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    npz_path = out / "similarity_100k.npz"
    np.savez_compressed(str(npz_path), ids=ids, vecs=vecs)
    np.save(str(out / "vectors_9d.npy"), vecs)
    return npz_path
