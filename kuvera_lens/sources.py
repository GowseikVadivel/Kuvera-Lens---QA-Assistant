"""Explicit official-source URL registry and validation."""

import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from kuvera_lens.schemes import SCHEME_IDS


OFFICIAL_HOSTS = {
    "hdfc_amc": frozenset(
        {"hdfcfund.com", "www.hdfcfund.com", "files.hdfcfund.com"}
    ),
    "sebi": frozenset({"sebi.gov.in", "www.sebi.gov.in"}),
    "amfi": frozenset({"amfiindia.com", "www.amfiindia.com"}),
}
_SOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$")


@dataclass(frozen=True)
class ApprovedSource:
    source_id: str
    scheme_id: str
    organization: str
    url: str


def validate_source(source: ApprovedSource) -> None:
    if not _SOURCE_ID_PATTERN.fullmatch(source.source_id):
        raise ValueError(f"Invalid source ID: {source.source_id!r}")
    if source.scheme_id not in SCHEME_IDS:
        raise ValueError(f"Unsupported scheme ID: {source.scheme_id!r}")
    if source.organization not in OFFICIAL_HOSTS:
        raise ValueError(f"Unsupported official organization: {source.organization!r}")

    parsed = urlsplit(source.url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Source URLs must use HTTPS and include a hostname.")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Source URLs must not contain credentials or fragments.")
    if parsed.port not in (None, 443):
        raise ValueError("Source URLs must use the default HTTPS port.")

    hostname = parsed.hostname.lower().rstrip(".")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("Source URLs must use an approved official hostname.")

    if hostname not in OFFICIAL_HOSTS[source.organization]:
        raise ValueError(
            f"Hostname {hostname!r} is not approved for {source.organization}."
        )


def load_sources(path: Path) -> list[ApprovedSource]:
    """Load only explicitly configured URLs; page links are never discovered here."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Source registry not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in source registry {path}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError("Source registry must contain a 'sources' list.")

    sources: list[ApprovedSource] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for index, item in enumerate(payload["sources"]):
        if not isinstance(item, dict):
            raise ValueError(f"Source entry {index} must be an object.")
        try:
            source = ApprovedSource(
                source_id=item["source_id"],
                scheme_id=item["scheme_id"],
                organization=item["organization"],
                url=item["url"],
            )
        except KeyError as exc:
            raise ValueError(
                f"Source entry {index} is missing required field {exc.args[0]!r}."
            ) from exc
        if not all(isinstance(value, str) for value in (
            source.source_id,
            source.scheme_id,
            source.organization,
            source.url,
        )):
            raise ValueError(f"Source entry {index} fields must all be strings.")
        validate_source(source)
        if source.source_id in seen_ids:
            raise ValueError(f"Duplicate source ID: {source.source_id!r}")
        if source.url in seen_urls:
            raise ValueError(f"Duplicate source URL: {source.url!r}")
        seen_ids.add(source.source_id)
        seen_urls.add(source.url)
        sources.append(source)

    return sources
