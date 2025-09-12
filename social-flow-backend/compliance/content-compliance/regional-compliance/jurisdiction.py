"""
jurisdiction.py

Utilities and models for jurisdiction resolution:
- map user locale, IP, or declared jurisdiction to canonical jurisdiction codes.
- simple heuristics, pluggable for IP geolocation services.
"""

from dataclasses import dataclass
from typing import Optional
import pycountry
import logging

logger = logging.getLogger(__name__)

# Canonical list of jurisdictions used by policies (short codes)
SUPPORTED_JURISDICTIONS = {"global", "us", "eu", "uk", "india", "ca", "au"}

@dataclass(frozen=True)
class JurisdictionInfo:
    code: str         # "us", "eu", "india"
    name: str
    region: Optional[str] = None

def normalize_locale_to_jurisdiction(locale: str) -> str:
    """
    Map locale strings (en-US, en_GB, fr-FR) to jurisdictions.
    Falls back to 'global'.
    """
    if not locale:
        return "global"
    code = locale.split("-")[-1].lower()
    # some normalization
    if code == "gb":
        code = "uk"
    if code in SUPPORTED_JURISDICTIONS:
        logger.debug(f"Locale {locale} mapped to jurisdiction {code}")
        return code
    # use pycountry to detect country alpha2 -> alpha2 mapping
    try:
        country = pycountry.countries.get(alpha_2=code.upper())
        if country:
            cc = country.alpha_2.lower()
            if cc in SUPPORTED_JURISDICTIONS:
                return cc
    except Exception:
        pass
    return "global"

def resolve_jurisdiction_from_user(user_profile: dict, ip_geolocation: Optional[dict] = None) -> str:
    """
    Heuristic resolution order:
    1) User-declared jurisdiction in profile
    2) profile locale
    3) IP geolocation country
    4) fallback to 'global'
    """
    if not user_profile:
        return "global"
    if user_profile.get("jurisdiction"):
        return user_profile["jurisdiction"].lower()
    locale = user_profile.get("locale")
    j = normalize_locale_to_jurisdiction(locale)
    if j != "global":
        return j
    if ip_geolocation and ip_geolocation.get("country_code"):
        cc = ip_geolocation["country_code"].lower()
        if cc == "gb":
            cc = "uk"
        if cc in SUPPORTED_JURISDICTIONS:
            return cc
    return "global"
