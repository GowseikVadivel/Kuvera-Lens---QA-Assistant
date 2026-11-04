"""Supported mutual fund schemes and their stable identifiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scheme:
    scheme_id: str
    display_name: str
    aliases: tuple[str, ...]


SUPPORTED_SCHEMES = (
    Scheme(
        "hdfc_large_cap_direct_growth",
        "HDFC Large Cap Fund – Direct Growth",
        ("HDFC Large Cap Fund", "HDFC Large Cap"),
    ),
    Scheme(
        "hdfc_flexi_cap_direct_growth",
        "HDFC Flexi Cap Fund – Direct Plan Growth",
        ("HDFC Flexi Cap Fund", "HDFC Flexi Cap"),
    ),
    Scheme(
        "hdfc_elss_tax_saver_direct_growth",
        "HDFC ELSS Tax Saver Fund – Direct Plan Growth",
        ("HDFC ELSS Tax Saver Fund", "HDFC ELSS Tax Saver"),
    ),
    Scheme(
        "hdfc_mid_cap_direct_growth",
        "HDFC Mid Cap Fund – Direct Growth",
        ("HDFC Mid Cap Fund", "HDFC Mid-Cap Fund"),
    ),
)

SCHEME_BY_ID = {scheme.scheme_id: scheme for scheme in SUPPORTED_SCHEMES}
SCHEME_IDS = frozenset(scheme.scheme_id for scheme in SUPPORTED_SCHEMES)
