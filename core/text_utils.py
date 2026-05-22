"""Text utility functions for city name normalization."""

CITY_ALIASES = {
    "ny": "newyork",
    "new york": "newyork",
    "new york city": "newyork",
    "la": "losangeles",
    "sf": "sanfrancisco",
    "lv": "lasvegas",
    "dc": "washington",
}


def normalize_city(s: str) -> str:
    """Normalize a city name by stripping, lowercasing, removing spaces,
    and resolving common aliases (e.g. 'New York' -> 'newyork')."""
    return CITY_ALIASES.get(
        s.strip().lower().replace(" ", ""),
        s.strip().lower().replace(" ", ""),
    )
