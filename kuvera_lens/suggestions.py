"""Fast, scheme-scoped suggested questions grounded in indexed source text."""

import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from qdrant_client import models

from kuvera_lens.retrieval import (
    ACTIVE_COLLECTION_ALIAS,
    RetrievalClient,
    SourceType,
    get_retrieval_client,
)
from kuvera_lens.schemes import SCHEME_BY_ID, SCHEME_IDS
from kuvera_lens.sources import OFFICIAL_HOSTS, ApprovedSource, validate_source

MAX_SUGGESTED_QUESTIONS = 3
SCHEME_SCROLL_LIMIT = 512


class SuggestionClient(RetrievalClient, Protocol):
    def scroll(
        self,
        collection_name: str,
        *,
        scroll_filter: models.Filter,
        limit: int,
        offset: object | None,
        with_payload: list[str] | bool,
        with_vectors: bool,
    ) -> tuple[list[object], object | None]:
        ...


@dataclass(frozen=True)
class SuggestedQuestion:
    question: str
    source_type: SourceType


@dataclass(frozen=True)
class _QuestionCandidate:
    template: str
    evidence_pattern: re.Pattern[str]
    source_priority: tuple[SourceType, ...]


_QUESTION_CANDIDATES = (
    _QuestionCandidate(
        "What is the minimum SIP in {scheme}?",
        re.compile(r"\bmin(?:imum)?\s+sip\b.{0,50}(?:\d|₹|rs\.?)", re.IGNORECASE),
        ("webpage", "sid_pdf"),
    ),
    _QuestionCandidate(
        "What is the exit load of {scheme}?",
        re.compile(
            r"\bexit\s+load\b.{0,350}?(?:\b(?:nil|no exit load)\b|\b\d+(?:\.\d+)?\s*%)",
            re.IGNORECASE | re.DOTALL,
        ),
        ("sid_pdf", "webpage"),
    ),
    _QuestionCandidate(
        "What is the investment objective of {scheme}?",
        re.compile(
            r"(?:investment objective.{0,200}(?:generate|provide|capital appreciation|income)"
            r"|aims? to (?:generate|provide))",
            re.IGNORECASE | re.DOTALL,
        ),
        ("sid_pdf", "webpage"),
    ),
    _QuestionCandidate(
        "What is the riskometer classification of {scheme}?",
        re.compile(r"\briskometer\b.{0,80}\b(?:low|moderate|high|very high)\b", re.IGNORECASE),
        ("webpage", "sid_pdf"),
    ),
    _QuestionCandidate(
        "What benchmark does {scheme} use?",
        re.compile(r"\bbenchmark\b.{0,100}\b(?:index|tri|nifty|bse|sensex)\b", re.IGNORECASE),
        ("webpage", "sid_pdf"),
    ),
    _QuestionCandidate(
        "What is the lock-in period for {scheme}?",
        re.compile(r"\block[- ]in\b.{0,80}\b(?:\d+\s*(?:years?|months?)|no lock)", re.IGNORECASE),
        ("webpage", "sid_pdf"),
    ),
)


def _validated_payload(payload: object, scheme_id: str) -> tuple[str, str] | None:
    if not isinstance(payload, dict) or payload.get("scheme_id") != scheme_id:
        return None

    document_type = payload.get("document_type")
    if document_type not in {"scheme_page", "SID"}:
        return None
    text = payload.get("text")
    source_id = payload.get("source_id")
    source_url = payload.get("source_url")
    source_domain = payload.get("source_domain")
    title = payload.get("document_title")
    content_hash = payload.get("content_hash")
    document_version = payload.get("document_version")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (text, source_id, source_url, source_domain, title, content_hash)
    ):
        return None
    if document_version != content_hash or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        return None

    try:
        parsed = urlsplit(source_url)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname != source_domain.lower().rstrip("."):
        return None
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
                scheme_id=scheme_id,
                organization=organization,
                url=source_url,
            )
        )
    except ValueError:
        return None

    source_type: SourceType = "webpage" if document_type == "scheme_page" else "sid_pdf"
    return text, source_type


def _scheme_source_texts(client: SuggestionClient, scheme_id: str) -> list[tuple[str, SourceType]]:
    points: list[object] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=ACTIVE_COLLECTION_ALIAS,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="scheme_id",
                        match=models.MatchValue(value=scheme_id),
                    )
                ]
            ),
            limit=SCHEME_SCROLL_LIMIT,
            offset=offset,
            with_payload=[
                "scheme_id",
                "document_type",
                "text",
                "source_id",
                "source_url",
                "source_domain",
                "document_title",
                "document_version",
                "content_hash",
            ],
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    source_texts = []
    for point in points:
        validated = _validated_payload(getattr(point, "payload", None), scheme_id)
        if validated is not None:
            source_texts.append(validated)
    return source_texts


def _build_suggested_questions(
    client: SuggestionClient,
    scheme_id: str,
) -> tuple[SuggestedQuestion, ...]:
    scheme = SCHEME_BY_ID[scheme_id]
    source_texts = _scheme_source_texts(client, scheme_id)
    supported: list[SuggestedQuestion] = []

    for candidate in _QUESTION_CANDIDATES:
        matched_sources = {
            source_type
            for text, source_type in source_texts
            if candidate.evidence_pattern.search(text)
        }
        source_type = next(
            (
                preferred
                for preferred in candidate.source_priority
                if preferred in matched_sources
            ),
            None,
        )
        if source_type is None:
            continue
        supported.append(
            SuggestedQuestion(
                candidate.template.format(scheme=scheme.display_name.removesuffix(" – Direct Growth").removesuffix(" – Direct Plan Growth")),
                source_type,
            )
        )
        if len(supported) == MAX_SUGGESTED_QUESTIONS:
            break
    return tuple(supported)


def get_suggested_questions(
    scheme_id: str,
    *,
    client: SuggestionClient | None = None,
) -> tuple[SuggestedQuestion, ...]:
    """Return evidence-supported questions for a scheme from one bounded Qdrant scan."""
    if scheme_id not in SCHEME_IDS:
        raise ValueError(f"Unsupported scheme ID: {scheme_id!r}")
    return _build_suggested_questions(client or get_retrieval_client(), scheme_id)
