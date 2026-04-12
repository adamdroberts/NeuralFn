"""Experimental: vocab-grounded semantic data layer for the Hybrid JEPA LLM.

Provides a vocab-grounded semantic space backed by shipped vocabulary
definitions, locality-sensitive hashing, and deterministic vocab-derived
semantic supervision. The default shipped router vocabulary is now 86 routed
dimensions plus one derived taxonomy-hash slot.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data" / "semantic"

DEFAULT_SEMANTIC_VOCAB_REF = "vocab_86d_o200k.json"
LEGACY_81D_SEMANTIC_VOCAB_REF = "vocab_81d.json"
LEGACY_SEMANTIC_VOCAB_REF = "vocab_8d.json"
TAXONOMY_HASH_NAME = "taxonomy_hash"
TAXONOMY_HASH_MEANING = "signature hash (entity+action+domain trigram)"
_SIGNATURE_DIMS = ("entity_type", "action", "domain")


class SemanticDim(NamedTuple):
    index: int
    name: str
    meaning: str


def resolve_semantic_vocab_path(path_or_ref: str | Path | None = None) -> Path:
    """Resolve a semantic vocab reference to an on-disk JSON path."""
    if path_or_ref in (None, ""):
        return _DATA_DIR / DEFAULT_SEMANTIC_VOCAB_REF
    candidate = Path(path_or_ref).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return _DATA_DIR / candidate.name


def normalize_semantic_vocab_ref(path_or_ref: str | Path | None = None) -> str:
    """Return a portable graph-storable semantic vocab reference."""
    if path_or_ref in (None, ""):
        return DEFAULT_SEMANTIC_VOCAB_REF
    path = resolve_semantic_vocab_path(path_or_ref)
    try:
        if path.parent.resolve() == _DATA_DIR.resolve():
            return path.name
    except FileNotFoundError:
        pass
    raw = Path(path_or_ref)
    if len(raw.parts) <= 1:
        return raw.name
    return str(raw)


def _load_vocab_payload(path_or_ref: str | Path | None = None) -> dict[str, Any]:
    path = resolve_semantic_vocab_path(path_or_ref)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Semantic vocabulary file {path} must contain a JSON object")
    raw_vocab = data.get("vocabulary")
    if not isinstance(raw_vocab, dict):
        raise ValueError(f"Semantic vocabulary file {path} is missing a valid 'vocabulary' object")
    return data


def semantic_dims_for_vocab(path_or_ref: str | Path | None = None) -> list[SemanticDim]:
    payload = _load_vocab_payload(path_or_ref)
    raw_vocab = payload["vocabulary"]
    descriptions = payload.get("dimension_descriptions")
    if not isinstance(descriptions, dict):
        descriptions = {}
    dims: list[SemanticDim] = []
    for idx, dim_name in enumerate(raw_vocab.keys()):
        meaning = descriptions.get(dim_name)
        if not isinstance(meaning, str) or not meaning.strip():
            meaning = dim_name.replace("_", " ")
        dims.append(SemanticDim(idx, dim_name, meaning))
    dims.append(SemanticDim(len(dims), TAXONOMY_HASH_NAME, TAXONOMY_HASH_MEANING))
    return dims


@lru_cache(maxsize=None)
def semantic_vector_dim_for_vocab(path_or_ref: str | Path | None = None) -> int:
    return len(semantic_dims_for_vocab(path_or_ref))


def semantic_vocab_ref_for_dim(semantic_dim: Any | None = None) -> str:
    """Resolve the shipped semantic vocab that matches a ref-less semantic shape."""
    try:
        dim = int(semantic_dim)
    except (TypeError, ValueError):
        return DEFAULT_SEMANTIC_VOCAB_REF

    if dim <= semantic_vector_dim_for_vocab(LEGACY_SEMANTIC_VOCAB_REF):
        return LEGACY_SEMANTIC_VOCAB_REF
    if dim == semantic_vector_dim_for_vocab(LEGACY_81D_SEMANTIC_VOCAB_REF):
        return LEGACY_81D_SEMANTIC_VOCAB_REF
    return DEFAULT_SEMANTIC_VOCAB_REF


def semantic_vocab_ref_for_graph(graph: Any) -> str:
    """Resolve the semantic vocab reference recorded on a graph, with legacy fallback."""
    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    template_spec = dict(torch_config.get("template_spec", {}) or {})
    artifact_metadata = dict(torch_config.get("artifact_metadata", {}) or {})
    for source in (template_spec, artifact_metadata):
        ref = source.get("semantic_vocab_ref")
        if isinstance(ref, str) and ref.strip():
            return normalize_semantic_vocab_ref(ref)

    sem_cfg: dict[str, Any] = {}
    try:
        sem_node = getattr(graph, "nodes", {}).get("semantic_data_source")
        if sem_node is not None:
            sem_cfg = dict(sem_node.neuron_def.module_config or {})
    except Exception:
        sem_cfg = {}
    ref = sem_cfg.get("semantic_vocab_ref")
    if isinstance(ref, str) and ref.strip():
        return normalize_semantic_vocab_ref(ref)

    semantic_dim = sem_cfg.get("seq_len", template_spec.get("semantic_dim"))
    return semantic_vocab_ref_for_dim(semantic_dim)


SEMANTIC_DIMS: list[SemanticDim] = semantic_dims_for_vocab(DEFAULT_SEMANTIC_VOCAB_REF)
SEMANTIC_DIM_NAMES: list[str] = [d.name for d in SEMANTIC_DIMS]
VOCAB_DIM_NAMES: list[str] = SEMANTIC_DIM_NAMES[:-1]
NUM_SEMANTIC_DIMS = len(SEMANTIC_DIM_NAMES)
NUM_VOCAB_DIMS = len(VOCAB_DIM_NAMES)
SEMANTIC_IGNORE_INDEX = -100
DEFAULT_SEMANTIC_SAMPLE_COUNT = 100_000
DIMENSION_TO_EXPERT_ID: dict[str, int] = {name: idx for idx, name in enumerate(VOCAB_DIM_NAMES)}
EXPERT_TO_DIMENSION: dict[int, str] = {idx: name for name, idx in DIMENSION_TO_EXPERT_ID.items()}


def _resolve_bucket_count(
    n_buckets: int = 4096,
    *,
    n_sig_buckets: int | None = None,
) -> int:
    if n_sig_buckets is not None:
        if n_buckets != 4096 and n_buckets != n_sig_buckets:
            raise ValueError("n_buckets and n_sig_buckets must match when both are provided")
        n_buckets = n_sig_buckets
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    return n_buckets


def signature_to_bucket(
    sig: str,
    n_buckets: int = 4096,
    *,
    n_sig_buckets: int | None = None,
) -> int:
    """Deterministically hash a semantic signature into a bucket id."""
    n_buckets = _resolve_bucket_count(n_buckets, n_sig_buckets=n_sig_buckets)
    h = int(hashlib.md5(sig.encode("utf-8")).hexdigest()[:8], 16)
    return h % n_buckets


def signature_to_float(
    sig: str,
    n_buckets: int = 4096,
    *,
    n_sig_buckets: int | None = None,
) -> float:
    """Deterministically hash a semantic_signature string to a float in [0, 1]."""
    n_buckets = _resolve_bucket_count(n_buckets, n_sig_buckets=n_sig_buckets)
    return signature_to_bucket(sig, n_buckets=n_buckets) / n_buckets


class ConversationalVocabulary:
    """Loads the shipped semantic topic vocabulary and term-index lookups."""

    def __init__(self, path: str | Path | None = None) -> None:
        resolved = resolve_semantic_vocab_path(path)
        data = _load_vocab_payload(resolved)
        raw_vocab = data["vocabulary"]
        dim_names = list(raw_vocab.keys())
        vocab: dict[str, list[str]] = {}
        for dim_name in dim_names:
            terms = raw_vocab[dim_name]
            if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
                raise ValueError(
                    f"Semantic vocabulary dimension {dim_name!r} must contain a list of string terms"
                )
            vocab[dim_name] = list(terms)
        term_counts = data.get("term_counts")
        if term_counts is not None:
            if not isinstance(term_counts, dict):
                raise ValueError("Semantic vocabulary 'term_counts' must be a JSON object when present")
            for dim_name in dim_names:
                expected = len(vocab[dim_name])
                actual = term_counts.get(dim_name)
                if actual != expected:
                    raise ValueError(
                        "Semantic vocabulary term_counts mismatch for "
                        f"{dim_name!r}: expected {expected}, got {actual!r}"
                    )
        total_terms = data.get("total_terms")
        if total_terms is not None:
            expected_total = sum(len(terms) for terms in vocab.values())
            if total_terms != expected_total:
                raise ValueError(
                    "Semantic vocabulary total_terms mismatch: "
                    f"expected {expected_total}, got {total_terms!r}"
                )
        self.path = resolved
        self.ref = normalize_semantic_vocab_ref(resolved)
        self.raw: dict[str, Any] = data
        self._vocab = vocab
        self._semantic_dims = semantic_dims_for_vocab(resolved)
        self._dimension_to_expert = {name: idx for idx, name in enumerate(self._vocab.keys())}
        self._term_index: dict[str, dict[str, int]] = {}
        for dim_name, terms in self._vocab.items():
            self._term_index[dim_name] = {t.lower(): i for i, t in enumerate(terms)}

    @property
    def dim_names(self) -> list[str]:
        return list(self._vocab.keys())

    @property
    def semantic_dims(self) -> list[SemanticDim]:
        return list(self._semantic_dims)

    @property
    def semantic_dim_names(self) -> list[str]:
        return [d.name for d in self._semantic_dims]

    @property
    def num_vocab_dims(self) -> int:
        return len(self._vocab)

    @property
    def vector_dim(self) -> int:
        return self.num_vocab_dims + 1

    def terms(self, dim_name: str) -> list[str]:
        return list(self._vocab[dim_name])

    @property
    def term_counts(self) -> dict[str, int]:
        return {dim_name: len(terms) for dim_name, terms in self._vocab.items()}

    @property
    def max_terms(self) -> int:
        return max((len(terms) for terms in self._vocab.values()), default=0)

    @property
    def dimension_to_expert(self) -> dict[str, int]:
        return dict(self._dimension_to_expert)

    @property
    def expert_to_dimension(self) -> dict[int, str]:
        return {idx: name for name, idx in self._dimension_to_expert.items()}

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
        """Encode a dimension/topic row dict to a float32 semantic vector."""
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        for i, dim_name in enumerate(self.dim_names):
            term = row.get(dim_name, "")
            idx = self.term_to_index(dim_name, term)
            n_terms = len(self._vocab.get(dim_name, []))
            if idx >= 0 and n_terms > 1:
                vec[i] = 2.0 * idx / (n_terms - 1) - 1.0
        sig = row.get("semantic_signature", "")
        if sig:
            vec[self.num_vocab_dims] = signature_to_float(sig, n_sig_buckets)
        return vec

    def decode_vector(self, vec: np.ndarray) -> dict[str, str | float]:
        """Map a semantic vector back to nearest vocabulary terms + hash scalar."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        result: dict[str, str | float] = {}
        for i, dim_name in enumerate(self.dim_names):
            terms = self._vocab.get(dim_name, [])
            if not terms:
                result[dim_name] = ""
                continue
            idx = int(round((vec[i] + 1.0) / 2.0 * (len(terms) - 1)))
            idx = max(0, min(idx, len(terms) - 1))
            result[dim_name] = terms[idx]
        if len(vec) > self.num_vocab_dims:
            result[TAXONOMY_HASH_NAME] = float(vec[self.num_vocab_dims])
        return result


def build_semantic_signature(
    target: np.ndarray,
    *,
    vocab: ConversationalVocabulary | None = None,
) -> str:
    """Build a stable signature string from entity/action/domain topic ids."""
    if vocab is None:
        vocab = ConversationalVocabulary()
    target = np.asarray(target, dtype=np.int64).ravel()
    parts: list[str] = []
    for dim_name in _SIGNATURE_DIMS:
        dim_idx = vocab.dimension_to_expert.get(dim_name)
        if dim_idx is None:
            parts.append("non")
            continue
        term_idx = int(target[dim_idx]) if dim_idx < target.shape[0] else SEMANTIC_IGNORE_INDEX
        terms = vocab.terms(dim_name)
        if 0 <= term_idx < len(terms):
            parts.append(terms[term_idx][:3].lower())
        else:
            parts.append("non")
    return "_".join(parts)


def build_semantic_targets_from_topics(
    topics: dict[str, str],
    *,
    vocab: ConversationalVocabulary | None = None,
    n_sig_buckets: int = 4096,
) -> np.ndarray:
    """Resolve a dimension/topic map to categorical semantic targets."""
    if vocab is None:
        vocab = ConversationalVocabulary()
    target = np.full(vocab.vector_dim, SEMANTIC_IGNORE_INDEX, dtype=np.int64)
    for dim_name, topic in topics.items():
        if dim_name not in vocab.dimension_to_expert:
            raise ValueError(f"Unknown semantic dimension {dim_name!r}")
        topic_idx = vocab.term_to_index(dim_name, topic)
        if topic_idx < 0:
            raise ValueError(f"Unknown topic {topic!r} for dimension {dim_name!r}")
        target[vocab.dimension_to_expert[dim_name]] = topic_idx
    target[vocab.num_vocab_dims] = signature_to_bucket(
        build_semantic_signature(target, vocab=vocab),
        n_buckets=n_sig_buckets,
    )
    return target


def semantic_targets_to_vectors(
    targets: np.ndarray,
    *,
    vocab: ConversationalVocabulary | None = None,
    n_sig_buckets: int = 4096,
) -> np.ndarray:
    """Convert categorical semantic targets into internal float semantic vectors."""
    if vocab is None:
        vocab = ConversationalVocabulary()
    arr = np.asarray(targets, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    vecs = np.zeros((arr.shape[0], vocab.vector_dim), dtype=np.float32)
    for dim_idx, dim_name in enumerate(vocab.dim_names):
        n_terms = len(vocab.terms(dim_name))
        indices = arr[:, dim_idx]
        valid = indices >= 0
        if n_terms > 1:
            vecs[valid, dim_idx] = 2.0 * indices[valid].astype(np.float32) / float(n_terms - 1) - 1.0
    if arr.shape[1] > vocab.num_vocab_dims:
        hash_indices = arr[:, vocab.num_vocab_dims]
        valid_hash = hash_indices >= 0
        if n_sig_buckets > 1:
            vecs[valid_hash, vocab.num_vocab_dims] = (
                hash_indices[valid_hash].astype(np.float32) / float(n_sig_buckets - 1)
            )
    return vecs


def semantic_targets_to_router_vectors(
    targets: np.ndarray,
    *,
    vocab: ConversationalVocabulary | None = None,
) -> np.ndarray:
    """Convert categorical semantic targets into 0..1 router vectors without the hash slot."""
    if vocab is None:
        vocab = ConversationalVocabulary()
    arr = np.asarray(targets, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    vecs = np.zeros((arr.shape[0], vocab.num_vocab_dims), dtype=np.float32)
    for dim_idx, dim_name in enumerate(vocab.dim_names):
        n_terms = len(vocab.terms(dim_name))
        indices = arr[:, dim_idx]
        valid = indices >= 0
        if n_terms > 1:
            vecs[valid, dim_idx] = indices[valid].astype(np.float32) / float(n_terms - 1)
        elif n_terms == 1:
            vecs[valid, dim_idx] = 1.0
    return vecs


def load_training_targets(
    n: int = DEFAULT_SEMANTIC_SAMPLE_COUNT,
    *,
    vocab: ConversationalVocabulary | None = None,
    seed: int = 0,
    active_dims: int = 2,
    n_sig_buckets: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize deterministic vocab-derived categorical semantic targets."""
    if vocab is None:
        vocab = ConversationalVocabulary()
    rng = np.random.RandomState(seed)
    ids = np.arange(n, dtype=np.int64)
    targets = np.full((n, vocab.vector_dim), SEMANTIC_IGNORE_INDEX, dtype=np.int64)
    active_count = max(1, min(int(active_dims), vocab.num_vocab_dims))
    for row_idx in range(n):
        chosen_dims = np.sort(rng.choice(vocab.num_vocab_dims, size=active_count, replace=False))
        for dim_idx in chosen_dims:
            dim_name = vocab.dim_names[int(dim_idx)]
            n_terms = len(vocab.terms(dim_name))
            topic_idx = int(rng.randint(0, max(n_terms, 1))) if n_terms > 0 else 0
            targets[row_idx, int(dim_idx)] = topic_idx
        sig = build_semantic_signature(targets[row_idx], vocab=vocab)
        targets[row_idx, vocab.num_vocab_dims] = signature_to_bucket(sig, n_buckets=n_sig_buckets)
    return ids, targets


def load_training_data(
    path: str | Path | None = None,
    vocab: ConversationalVocabulary | None = None,
    n_sig_buckets: int = 4096,
    *,
    n: int = DEFAULT_SEMANTIC_SAMPLE_COUNT,
    seed: int = 0,
    active_dims: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning deterministic vocab-derived semantic vectors."""
    del path
    ids, targets = load_training_targets(
        n=n,
        vocab=vocab,
        seed=seed,
        active_dims=active_dims,
        n_sig_buckets=n_sig_buckets,
    )
    vecs = semantic_targets_to_vectors(targets, vocab=vocab, n_sig_buckets=n_sig_buckets)
    return ids, vecs


class SemanticMatrix:
    """In-memory semantic similarity matrix."""

    def __init__(self, path: str | Path) -> None:
        data = np.load(str(path), allow_pickle=True)
        self.ids: np.ndarray = data["ids"]
        self.vectors: np.ndarray = data["vecs"].astype(np.float32)
        if self.vectors.ndim != 2 or self.vectors.shape[1] <= 0:
            raise ValueError("SemanticMatrix expects 2D float vectors with a non-zero feature dimension")
        self.dim = int(self.vectors.shape[1])
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.vectors = self.vectors / norms

    def neighbors(self, vec: np.ndarray, k: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) for the *k* nearest neighbours by cosine."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim}-D query, got {vec.shape[0]}-D")
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
        hasher = SemanticHasher(dim=self.dim, tables=tables, planes=planes, seed=seed)
        hasher.index(self.ids, self.vectors)
        return hasher


class SemanticHasher:
    """Multi-table random-hyperplane LSH for semantic vectors."""

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
        self._buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(tables)]
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

    try:
        vocab = ConversationalVocabulary()
        ids, targets = load_training_targets(n=n, vocab=vocab, seed=seed, active_dims=min(2, vocab.num_vocab_dims))
        vecs = semantic_targets_to_vectors(targets, vocab=vocab)
    except (FileNotFoundError, json.JSONDecodeError):
        rng = np.random.RandomState(seed)
        ids = np.arange(n, dtype=np.int64)
        vecs = rng.randn(n, dim).astype(np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    npz_path = out / "similarity_100k.npz"
    np.savez_compressed(str(npz_path), ids=ids, vecs=vecs)
    np.save(str(out / f"vectors_{vecs.shape[1]}d.npy"), vecs)
    return npz_path
