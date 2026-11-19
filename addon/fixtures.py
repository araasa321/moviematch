"""Synthetic fixture catalog + Trakt history.

Deliberately trivial vectors so the *math* is auditable by hand:
    Batman-like   -> e_y  (The Dark Knight, Batman Begins, TDK Rises, Joker)
    Nolan-puzzle  -> e_x  (Inception, The Prestige, Memento, Breaking Bad)
    space         -> e_z  (Interstellar, 2001, Spirited Away)
Titles 155/272/49026 share TMDB collection 263 so the collection rule bites.
Two titles carry a non-English language so the language penalty is exercised.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from .ids import point_id

BATMAN = (0.0, 1.0, 0.0)
NOLAN = (1.0, 0.0, 0.0)
SPACE = (0.0, 0.0, 1.0)

# Genres per tmdb id. Real TMDB payloads always carry a genre list; the fixture
# catalog used to leave it empty, which made every genre-filtering test vacuous.
# Names match `policy.GENRE_NAMES` (i.e. what the ingest job actually stores).
GENRES: dict[int, tuple[str, ...]] = {
    155: ("Action", "Crime", "Drama", "Thriller"),
    272: ("Action", "Crime", "Drama"),
    49026: ("Action", "Crime", "Drama", "Thriller"),
    475557: ("Crime", "Drama", "Thriller"),
    27205: ("Action", "Sci-Fi", "Thriller"),
    1124: ("Drama", "Mystery", "Thriller"),
    77: ("Mystery", "Thriller"),
    157336: ("Adventure", "Drama", "Sci-Fi"),
    62: ("Sci-Fi", "Adventure", "Mystery"),
    129: ("Animation", "Family", "Fantasy"),
    550: ("Drama", "Thriller"),
    1396: ("Crime", "Drama", "Thriller"),
}


def imdb_id(tmdb: int) -> str:
    return f"tt{1_000_000 + tmdb:07d}"


def title(media_type: str, tmdb_id: int, name: str, vector: Sequence[float], *, year: int,
          vote_average: float, vote_count: int, popularity: float,
          director: str = "", cast: Sequence[str] = (), language: str = "en",
          genres: Sequence[str] = (), collection_id: int | None = None,
          seasons: int = 1) -> dict[str, Any]:
    payload = {
        "tmdb_id": tmdb_id,
        "media_type": media_type,
        "imdb_id": imdb_id(tmdb_id),
        "title": name,
        "year": year,
        "end_year": None,
        "genres": list(genres),
        "keywords": [],
        "director": director,
        "cast": list(cast),
        "language": language,
        "runtime": 120 if media_type == "movie" else None,
        "vote_average": vote_average,
        "vote_count": vote_count,
        "popularity": popularity,
        "budget": 0,
        "revenue": 0,
        "collection_id": collection_id,
        "release_date": f"{year}-07-16",
        "poster": f"/poster-{tmdb_id}.jpg",
        "backdrop": f"/backdrop-{tmdb_id}.jpg",
        "overview_short": f"{name} overview.",
        "seasons": seasons if media_type == "series" else None,
        "enriched": True,
    }
    return {"point_id": point_id(media_type, tmdb_id), "vector": list(vector), "payload": payload}


def catalog(dim: int = 3) -> list[dict[str, Any]]:
    def vec(base: Sequence[float]) -> list[float]:
        return list(base) + [0.0] * (dim - 3)

    rows: list[tuple] = [
        (155, "The Dark Knight", BATMAN, 2008, 8.5, 32000, 120.4,
         "Christopher Nolan", ["Christian Bale", "Heath Ledger"], 263, "en", "movie"),
        (272, "Batman Begins", BATMAN, 2005, 7.7, 21000, 90.1,
         "Christopher Nolan", ["Christian Bale", "Liam Neeson"], 263, "en", "movie"),
        (49026, "The Dark Knight Rises", BATMAN, 2012, 7.8, 18000, 100.2,
         "Christopher Nolan", ["Christian Bale", "Tom Hardy"], 263, "en", "movie"),
        (475557, "Joker", BATMAN, 2019, 8.2, 26000, 150.0,
         "Todd Phillips", ["Joaquin Phoenix"], None, "en", "movie"),
        (27205, "Inception", NOLAN, 2010, 8.4, 36000, 160.7,
         "Christopher Nolan", ["Leonardo DiCaprio", "Tom Hardy"], None, "en", "movie"),
        (1124, "The Prestige", NOLAN, 2006, 8.2, 15000, 70.3,
         "Christopher Nolan", ["Christian Bale", "Hugh Jackman"], None, "en", "movie"),
        (77, "Memento", NOLAN, 2000, 8.2, 14000, 60.4,
         "Christopher Nolan", ["Guy Pearce"], None, "en", "movie"),
        (157336, "Interstellar", SPACE, 2014, 8.4, 33000, 140.9,
         "Christopher Nolan", ["Matthew McConaughey"], None, "en", "movie"),
        (62, "2001: A Space Odyssey", SPACE, 1968, 8.1, 11000, 40.2,
         "Stanley Kubrick", ["Keir Dullea"], None, "en", "movie"),
        (129, "Spirited Away", SPACE, 2001, 8.6, 16000, 55.5,
         "Hayao Miyazaki", ["Rumi Hiiragi"], None, "ja", "movie"),
        (550, "Fight Club", BATMAN, 1999, 8.4, 25000, 80.0,
         "David Fincher", ["Brad Pitt"], None, "id", "movie"),
        (1396, "Breaking Bad", NOLAN, 2008, 8.9, 14000, 95.0,
         "Vince Gilligan", ["Bryan Cranston", "Aaron Paul"], None, "en", "series"),
    ]
    return [
        title(media_type, tmdb, name, vec(base), year=year, vote_average=avg,
              vote_count=votes, popularity=pop, director=director, cast=cast,
              collection_id=coll, language=language,
              genres=GENRES.get(tmdb, ("Drama",)))
        for tmdb, name, base, year, avg, votes, pop, director, cast, coll, language, media_type in rows
    ]


def trakt_history(now: datetime | None = None) -> dict[str, Any]:
    """Trakt-shaped payloads. Includes one item with no tmdb id (must be skipped)."""
    now = now or datetime.now(timezone.utc)
    day = timedelta(days=1)
    return {
        "watched": [
            {"movie": {"ids": {"tmdb": 155}}, "plays": 3, "watched_at": (now - 30 * day).isoformat()},
            {"show": {"ids": {"tmdb": 1396}}, "plays": 1, "watched_at": (now - 10 * day).isoformat()},
            {"movie": {"ids": {"trakt": 1}}, "plays": 1, "watched_at": (now - 5 * day).isoformat()},
        ],
        "ratings": [
            {"movie": {"ids": {"tmdb": 27205}}, "rating": 10, "rated_at": (now - 2 * day).isoformat()},
            {"show": {"ids": {"tmdb": 1396}}, "rating": 9, "rated_at": (now - 10 * day).isoformat()},
            {"movie": {"ids": {"tmdb": 550}}, "rating": 2, "rated_at": (now - 4 * day).isoformat()},
        ],
        "watchlist": [
            {"movie": {"ids": {"tmdb": 1124}}, "listed_at": (now - 1 * day).isoformat()},
        ],
    }
