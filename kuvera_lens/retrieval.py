"""Scheme-filtered retrieval and provenance validation for indexed evidence."""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol
from urllib.parse import urlsplit

import numpy as np
from qdrant_client import QdrantClient, models

from kuvera_lens.embedding import EmbeddingModel, _load_model
from kuvera_lens.qdrant_store import ACTIVE_COLLECTION_ALIAS, create_qdrant_client
from kuvera_lens.schemes import SCHEME_IDS
from kuvera_lens.sources import OFFICIAL_HOSTS, ApprovedSource, validate_source

DEFAULT_TOP_K = 5
DEFAULT_MINIMUM_COSINE_SIMILARITY = 0.35
SOURCE_FILTER_CANDIDATE_LIMIT = 2048
SourceType = Literal["webpage", "sid_pdf", "both"]
_DOCUMENT_TYPE_BY_SOURCE_TYPE = {
    "webpage": ("scheme_page",),
    "sid_pdf": ("SID",),
    "both": ("scheme_page", "SID"),
}
_RETRIEVAL_QUERY_REWRITES = (
    (re.compile(r"\b(?:expense\s+ratio|TER)\b", re.IGNORECASE), "What TER does this mutual fund charge?"),
    (re.compile(r"\briskometer\b", re.IGNORECASE), "What is the scheme Riskometer level?"),
    (
        re.compile(r"\b(?:asset\s+allocation|allocation\s+limits?)\b", re.IGNORECASE),
        "What are the indicative asset allocations and minimum and maximum percentages by instrument?",
    ),
    (
        re.compile(r"\b(?:principal\s+)?risks?\b", re.IGNORECASE),
        "What are the principal risks and risk factors associated with the scheme's investments?",
    ),
)
VERIFICATION_REFUSAL = (
    "I couldn't verify this information from the available official sources."
)
_REQUIRED_METADATA = (
    "chunk_id",
    "source_id",
    "scheme_id",
    "scheme_name",
    "source_url",
    "source_domain",
    "document_title",
    "document_type",
    "document_version",
    "content_hash",
)


class UnsupportedSchemeError(ValueError):
    """The selected scheme is not one of the configured schemes."""


class RetrievalClient(Protocol):
    def query_points(
        self,
        collection_name: str,
        query: list[float],
        *,
        query_filter: models.Filter,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
        score_threshold: float | None,
    ) -> object:
        ...


@dataclass(frozen=True)
class Citation:
    source_id: str
    title: str
    source_url: str
    page_number: int | None
    section: str | None


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    scheme_id: str
    score: float
    citation: Citation


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[Evidence, ...]
    refusal: str | None = None

    @property
    def is_refusal(self) -> bool:
        return self.refusal is not None


def get_retrieval_client() -> QdrantClient:
    """Return a client for the shared active alias, or isolated local dev store."""
    return create_qdrant_client()


@lru_cache(maxsize=1)
def _load_retrieval_model() -> EmbeddingModel:
    return _load_model()


def _validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    return question.strip()


def _retrieval_query(question: str) -> str:
    for pattern, rewrite in _RETRIEVAL_QUERY_REWRITES:
        if pattern.search(question):
            return rewrite
    return question


def _has_direct_field_match(
    question: str,
    text: str,
    document_type: str,
) -> bool:
    if document_type == "SID":
        if not re.search(
            r"\b(?:asset\s+allocation|allocation\s+limits?)\b",
            question,
            re.IGNORECASE,
        ):
            return False
        return bool(
            re.search(r"\bIndicative allocations\b", text, re.IGNORECASE)
            and re.search(r"\bMinimum\b", text, re.IGNORECASE)
            and re.search(r"\bMaximum\b", text, re.IGNORECASE)
            and re.search(
                r"\bEquity and Equity Related Instruments\b",
                text,
                re.IGNORECASE,
            )
        )
    if document_type != "scheme_page":
        return False
    if re.search(r"\b(?:expense\s+ratio|TER)\b", question, re.IGNORECASE):
        return bool(re.search(r"\bTER\b\s*\d+(?:\.\d+)?%?(?![\d.])", text, re.IGNORECASE))
    if re.search(r"\briskometer\b", question, re.IGNORECASE):
        return bool(
            re.search(
                r"\bRiskometer\b\s*(?:[:#-]\s*)?\b"
                r"(?:very\s+high|moderately\s+high|high|moderate|low)\b",
                text,
                re.IGNORECASE,
            )
        )
    return False


def _is_kim_document(title: str) -> bool:
    return bool(
        re.search(
            r"\b(?:key information memorandum|key information document|KIM)\b",
            title,
            re.IGNORECASE,
        )
    )


def _normalize_query_vector(
    question: str,
    model: EmbeddingModel | None,
) -> list[float]:
    encoder = model or _load_retrieval_model()
    encoded = np.asarray(
        encoder.encode(
            [question],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] < 1:
        raise RuntimeError("Embedding model returned an invalid question vector shape.")
    vector = encoded[0]
    if not np.isfinite(vector).all():
        raise RuntimeError("Embedding model returned non-finite question vector values.")
    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise RuntimeError("Embedding model returned a non-normalized question vector.")
    return vector.tolist()


def _metadata_text(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _approved_evidence(
    point: object,
    *,
    question: str,
    selected_scheme_id: str,
    source_type: SourceType,
    minimum_score: float,
) -> Evidence | None:
    score = getattr(point, "score", None)
    payload = getattr(point, "payload", None)
    if (
        not isinstance(score, (int, float))
        or not np.isfinite(score)
        or not isinstance(payload, dict)
    ):
        return None
    text = _metadata_text(payload, "text")
    if not text:
        return None

    values = {key: _metadata_text(payload, key) for key in _REQUIRED_METADATA}
    if any(value is None for value in values.values()):
        return None
    if values["scheme_id"] != selected_scheme_id:
        return None
    if values["document_type"] not in _DOCUMENT_TYPE_BY_SOURCE_TYPE[source_type]:
        return None
    document_type = values["document_type"]
    title = values["document_title"] or ""
    if document_type == "SID" and _is_kim_document(title):
        return None
    if score < minimum_score and not _has_direct_field_match(
        question, text, document_type or ""
    ):
        return None

    source_url = values["source_url"]
    source_domain = values["source_domain"]
    source_id = values["source_id"]
    if source_url is None or source_domain is None or source_id is None:
        return None
    try:
        parsed = urlsplit(source_url)
    except ValueError:
        return None
    hostname = parsed.hostname
    if not hostname or hostname.lower().rstrip(".") != source_domain.lower().rstrip("."):
        return None
    hostname = hostname.lower().rstrip(".")
    organization = next(
        (name for name, hosts in OFFICIAL_HOSTS.items() if hostname in hosts),
        None,
    )
    if organization is None:
        return None
    try:
        validate_source(
            ApprovedSource(
                source_id=source_id,
                scheme_id=selected_scheme_id,
                organization=organization,
                url=source_url,
            )
        )
    except ValueError:
        return None

    content_hash = values["content_hash"]
    if content_hash is None or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        return None
    if values["document_version"] != content_hash:
        return None
    chunk_id = values["chunk_id"]
    if chunk_id is None:
        return None

    page_number_value = payload.get("page_number")
    page_number = (
        page_number_value
        if isinstance(page_number_value, int) and not isinstance(page_number_value, bool)
        else None
    )
    section = _metadata_text(payload, "section")
    return Evidence(
        chunk_id=chunk_id,
        text=text,
        scheme_id=selected_scheme_id,
        score=float(score),
        citation=Citation(
            source_id=source_id,
            title=title,
            source_url=source_url,
            page_number=page_number,
            section=section,
        ),
    )


def retrieve_evidence(
    question: str,
    scheme_id: str,
    *,
    client: RetrievalClient | None = None,
    model: EmbeddingModel | None = None,
    source_type: SourceType = "both",
    top_k: int = DEFAULT_TOP_K,
    minimum_similarity: float = DEFAULT_MINIMUM_COSINE_SIMILARITY,
) -> RetrievalResult:
    """Retrieve approved evidence filtered by scheme and source type."""
    if scheme_id not in SCHEME_IDS:
        raise UnsupportedSchemeError(f"Unsupported scheme ID: {scheme_id!r}")
    normalized_question = _validate_question(question)
    if not isinstance(source_type, str) or source_type not in _DOCUMENT_TYPE_BY_SOURCE_TYPE:
        raise ValueError(f"Unsupported source type: {source_type!r}")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")
    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be between zero and one.")

    query_vector = _normalize_query_vector(_retrieval_query(normalized_question), model)
    qdrant_client = client or get_retrieval_client()
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="scheme_id",
                match=models.MatchValue(value=scheme_id),
            )
        ]
    )
    response = qdrant_client.query_points(
        collection_name=ACTIVE_COLLECTION_ALIAS,
        query=query_vector,
        query_filter=query_filter,
        limit=SOURCE_FILTER_CANDIDATE_LIMIT,
        with_payload=True,
        with_vectors=False,
        score_threshold=None,
    )
    points = getattr(response, "points", None)
    if not isinstance(points, list):
        raise RuntimeError("Qdrant returned an invalid retrieval response.")

    evidence_by_type: dict[str, list[Evidence]] = {
        "scheme_page": [],
        "SID": [],
    }
    seen_chunk_ids: set[str] = set()
    for point in points:
        payload = getattr(point, "payload", None)
        document_type = (
            payload.get("document_type") if isinstance(payload, dict) else None
        )
        if not isinstance(document_type, str):
            continue
        item = _approved_evidence(
            point,
            question=normalized_question,
            selected_scheme_id=scheme_id,
            source_type=source_type,
            minimum_score=minimum_similarity,
        )
        if (
            item is None
            or item.chunk_id in seen_chunk_ids
            or document_type not in evidence_by_type
        ):
            continue
        evidence_by_type[document_type].append(item)
        seen_chunk_ids.add(item.chunk_id)

    for document_type, items in evidence_by_type.items():
        items.sort(
            key=lambda item: (
                not _has_direct_field_match(
                    normalized_question,
                    item.text,
                    document_type,
                ),
                -item.score,
            )
        )

    if source_type == "both":
        evidence = []
        if top_k > 1:
            for document_type in ("scheme_page", "SID"):
                if evidence_by_type[document_type]:
                    evidence.append(evidence_by_type[document_type][0])
        selected_ids = {item.chunk_id for item in evidence}
        remaining = sorted(
            (
                item
                for items in evidence_by_type.values()
                for item in items
                if item.chunk_id not in selected_ids
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        evidence.extend(remaining[: max(0, top_k - len(evidence))])
    else:
        document_type = _DOCUMENT_TYPE_BY_SOURCE_TYPE[source_type][0]
        evidence = evidence_by_type[document_type][:top_k]

    if not evidence:
        return RetrievalResult((), VERIFICATION_REFUSAL)
    return RetrievalResult(tuple(evidence))


__all__ = [
    "ACTIVE_COLLECTION_ALIAS",
    "Citation",
    "DEFAULT_MINIMUM_COSINE_SIMILARITY",
    "DEFAULT_TOP_K",
    "SOURCE_FILTER_CANDIDATE_LIMIT",
    "Evidence",
    "RetrievalResult",
    "SourceType",
    "UnsupportedSchemeError",
    "VERIFICATION_REFUSAL",
    "create_qdrant_client",
    "get_retrieval_client",
    "retrieve_evidence",
]
