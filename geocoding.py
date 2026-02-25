"""Geocoding utilities for location-based job search."""

import json
import time
from pathlib import Path

import requests

# Cache file for geocoding results
GEOCODE_CACHE_FILE = Path(".cache/geocode_cache.json")

# Rate limit: Nominatim requires max 1 request per second
_last_request_time: float = 0.0


def _load_cache() -> dict[str, tuple[float, float]]:
    """Load geocoding cache from disk."""
    if GEOCODE_CACHE_FILE.exists():
        try:
            with open(GEOCODE_CACHE_FILE) as f:
                data = json.load(f)
                # Convert lists back to tuples with explicit typing
                return {k: (float(v[0]), float(v[1])) for k, v in data.items()}
        except json.JSONDecodeError, OSError:
            pass
    return {}


def _save_cache(cache: dict[str, tuple[float, float]]) -> None:
    """Save geocoding cache to disk."""
    GEOCODE_CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(GEOCODE_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def geocode_city(
    city: str, state: str | None = None, country: str = "USA"
) -> tuple[float, float] | None:
    """Look up coordinates for a city using OpenStreetMap Nominatim.

    Args:
        city: City name.
        state: State/province name (optional).
        country: Country name (default: USA).

    Returns:
        Tuple of (latitude, longitude) or None if not found.
    """
    global _last_request_time

    # Build cache key
    cache_key = f"{city}|{state or ''}|{country}".lower()

    # Check cache first
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    # Build query string
    parts = [city]
    if state:
        parts.append(state)
    parts.append(country)
    query = ", ".join(parts)

    # Rate limiting: ensure at least 1 second between requests
    elapsed = time.time() - _last_request_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "JobSearchDemo/1.0"},
            timeout=10,
        )
        _last_request_time = time.time()

        if resp.status_code == 200:
            results = resp.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                # Cache the result
                cache[cache_key] = (lat, lon)
                _save_cache(cache)
                return (lat, lon)
    except requests.RequestException, KeyError, ValueError, IndexError:
        # Geocoding failed, return None
        pass

    return None


def geocode_with_fallback(
    city: str | None,
    state: str | None,
    llm_lat: float | None,
    llm_lon: float | None,
) -> tuple[float | None, float | None]:
    """Get coordinates with geocoding fallback.

    If LLM provided coordinates, use them. Otherwise, try geocoding.

    Args:
        city: City name from parsed intent.
        state: State name from parsed intent.
        llm_lat: Latitude from LLM (may be None or inaccurate).
        llm_lon: Longitude from LLM (may be None or inaccurate).

    Returns:
        Tuple of (latitude, longitude), either may be None.
    """
    # If LLM provided coordinates, use them
    if llm_lat is not None and llm_lon is not None:
        return (llm_lat, llm_lon)

    # If we have a city, try to geocode it
    if city:
        coords = geocode_city(city, state)
        if coords:
            return coords

    # No coordinates available
    return (None, None)
