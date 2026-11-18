"""Build a 200-title eval set with real (public) metadata.

Source: Wikipedia REST summaries -> exactly the fields PLAN §6.1 embeds:
        title, director, cast, overview. No TMDB token needed.

Writes eval/data/titles.json:
    [{"name", "year", "director", "cast": [...], "overview", "franchise", "note"}]
The `franchise` tag is used later to report *why* a title is similar
(franchise/same-director/theme/drama) — that is the core question of this eval.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data" / "titles.json"
UA = {"User-Agent": "MovieMatch-eval/0.1 (personal research; offline cache)"}
API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# name, year, wikipedia page, director, franchise, note
SEED: list[tuple[str, int, str, str, str, str]] = [
    # --- Batman / Nolan (the §15 acceptance seed) ---
    ("The Dark Knight", 2008, "The Dark Knight (film)", "Christopher Nolan", "batman", "SEED"),
    ("Batman Begins", 2005, "Batman Begins", "Christopher Nolan", "batman", "collection"),
    ("The Dark Knight Rises", 2012, "The Dark Knight Rises", "Christopher Nolan", "batman", "collection"),
    ("Joker", 2019, "Joker (2019 film)", "Todd Phillips", "batman", "same-world"),
    ("The Batman", 2022, "The Batman (film)", "Matt Reeves", "batman", "same-world"),
    ("Batman", 1989, "Batman (1989 film)", "Tim Burton", "batman", "same-world"),
    ("Batman Returns", 1992, "Batman Returns", "Tim Burton", "batman", "same-world"),
    ("Batman v Superman: Dawn of Justice", 2016, "Batman v Superman: Dawn of Justice", "Zack Snyder", "batman", "same-world"),
    ("The Lego Batman Movie", 2017, "The Lego Batman Movie", "Chris McKay", "batman", "same-world"),
    ("Joker: Folie a Deux", 2024, "Joker: Folie a Deux", "Todd Phillips", "batman", "same-world"),
    # --- target set from PLAN §15 ---
    ("Inception", 2010, "Inception", "Christopher Nolan", "nolan", "target"),
    ("The Prestige", 2006, "The Prestige (film)", "Christopher Nolan", "nolan", "target"),
    ("Heat", 1995, "Heat (1995 film)", "Michael Mann", "heist", "target"),
    # --- Nolan rest (same-director confound) ---
    ("Memento", 2000, "Memento (film)", "Christopher Nolan", "nolan", ""),
    ("Interstellar", 2014, "Interstellar (film)", "Christopher Nolan", "nolan", ""),
    ("Dunkirk", 2017, "Dunkirk (film)", "Christopher Nolan", "nolan", ""),
    ("Tenet", 2020, "Tenet (film)", "Christopher Nolan", "nolan", ""),
    ("Oppenheimer", 2023, "Oppenheimer (film)", "Christopher Nolan", "nolan", ""),
    ("Insomnia", 2002, "Insomnia (2002 film)", "Christopher Nolan", "nolan", ""),
    ("Following", 1998, "Following", "Christopher Nolan", "nolan", ""),
    # --- theme-blockbusters: crime epics, antihero, moral dilemma ---
    ("The Departed", 2006, "The Departed", "Martin Scorsese", "crime", "theme"),
    ("Goodfellas", 1990, "Goodfellas", "Martin Scorsese", "crime", "theme"),
    ("The Godfather", 1972, "The Godfather", "Francis Ford Coppola", "crime", "theme"),
    ("The Godfather Part II", 1974, "The Godfather Part II", "Francis Ford Coppola", "crime", "theme"),
    ("Casino", 1995, "Casino (film)", "Martin Scorsese", "crime", "theme"),
    ("Se7en", 1995, "Seven (1995 film)", "David Fincher", "crime", "theme"),
    ("Zodiac", 2007, "Zodiac (film)", "David Fincher", "crime", "theme"),
    ("Prisoners", 2013, "Prisoners (2013 film)", "Denis Villeneuve", "crime", "theme"),
    ("Sicario", 2015, "Sicario (2015 film)", "Denis Villeneuve", "crime", "theme"),
    ("Mystic River", 2003, "Mystic River (film)", "Clint Eastwood", "crime", "theme"),
    ("The Town", 2010, "The Town (2010 film)", "Ben Affleck", "heist", "theme"),
    ("Collateral", 2004, "Collateral (film)", "Michael Mann", "crime", "theme"),
    ("Nightcrawler", 2014, "Nightcrawler (film)", "Dan Gilroy", "crime", "theme"),
    ("Oldboy", 2003, "Oldboy (2003 film)", "Park Chan-wook", "crime", "theme"),
    ("L.A. Confidential", 1997, "L.A. Confidential (film)", "Curtis Hanson", "crime", "theme"),
    ("Michael Clayton", 2007, "Michael Clayton", "Tony Gilroy", "crime", ""),
    ("Road to Perdition", 2002, "Road to Perdition", "Sam Mendes", "crime", ""),
    ("American Psycho", 2000, "American Psycho (film)", "Mary Harron", "crime", ""),
    # --- target franchise neighbours / superhero peers ---
    ("Watchmen", 2009, "Watchmen (film)", "Zack Snyder", "superhero", "theme"),
    ("V for Vendetta", 2005, "V for Vendetta (film)", "James McTeigue", "superhero", "theme"),
    ("Unbreakable", 2000, "Unbreakable (film)", "M. Night Shyamalan", "superhero", "theme"),
    ("Logan", 2017, "Logan (film)", "James Mangold", "superhero", "theme"),
    ("X-Men: Days of Future Past", 2014, "X-Men: Days of Future Past", "Bryan Singer", "superhero", ""),
    ("Spider-Man 2", 2004, "Spider-Man 2", "Sam Raimi", "superhero", ""),
    ("The Amazing Spider-Man", 2012, "The Amazing Spider-Man (film)", "Marc Webb", "superhero", ""),
    ("Avengers: Endgame", 2019, "Avengers: Endgame", "Anthony and Joe Russo", "superhero", "popular"),
    ("Avengers: Infinity War", 2018, "Avengers: Infinity War", "Anthony and Joe Russo", "superhero", "popular"),
    ("Iron Man", 2008, "Iron Man (2008 film)", "Jon Favreau", "superhero", ""),
    ("Guardians of the Galaxy", 2014, "Guardians of the Galaxy (film)", "James Gunn", "superhero", ""),
    ("Deadpool", 2016, "Deadpool (film)", "Tim Miller", "superhero", ""),
    ("Man of Steel", 2013, "Man of Steel (film)", "Zack Snyder", "superhero", ""),
    ("Wonder Woman", 2017, "Wonder Woman (2017 film)", "Patty Jenkins", "superhero", ""),
    ("Black Panther", 2018, "Black Panther (film)", "Ryan Coogler", "superhero", ""),
    ("Superman", 1978, "Superman (1978 film)", "Richard Donner", "superhero", ""),
    ("The Incredibles", 2004, "The Incredibles", "Brad Bird", "animation", ""),
    ("Spider-Man: Into the Spider-Verse", 2018, "Spider-Man: Into the Spider-Verse", "Bob Persichetti", "animation", ""),
    # --- adjacent crime/thriller that should be *plausibly* good ---
    ("Kill Bill: Volume 1", 2003, "Kill Bill: Volume 1", "Quentin Tarantino", "tarantino", "adjacent"),
    ("Pulp Fiction", 1994, "Pulp Fiction", "Quentin Tarantino", "tarantino", "adjacent"),
    ("Reservoir Dogs", 1992, "Reservoir Dogs", "Quentin Tarantino", "tarantino", ""),
    ("Django Unchained", 2012, "Django Unchained", "Quentin Tarantino", "tarantino", ""),
    ("Once Upon a Time in Hollywood", 2019, "Once Upon a Time in Hollywood", "Quentin Tarantino", "tarantino", ""),
    ("Inglourious Basterds", 2009, "Inglourious Basterds", "Quentin Tarantino", "tarantino", ""),
    ("Drive", 2011, "Drive (2011 film)", "Nicolas Winding Refn", "neonoir", ""),
    ("Blade Runner 2049", 2017, "Blade Runner 2049", "Denis Villeneuve", "scifi", ""),
    ("Arrival", 2016, "Arrival (film)", "Denis Villeneuve", "scifi", ""),
    ("Dune", 2021, "Dune (2021 film)", "Denis Villeneuve", "scifi", ""),
    ("Dune: Part Two", 2024, "Dune: Part Two", "Denis Villeneuve", "scifi", ""),
    ("Blade Runner", 1982, "Blade Runner", "Ridley Scott", "scifi", ""),
    ("The Matrix", 1999, "The Matrix", "The Wachowskis", "scifi", ""),
    ("The Matrix Reloaded", 2003, "The Matrix Reloaded", "The Wachowskis", "scifi", ""),
    ("Ex Machina", 2014, "Ex Machina (film)", "Alex Garland", "scifi", ""),
    ("The Martian", 2015, "The Martian (film)", "Ridley Scott", "scifi", ""),
    ("Gravity", 2013, "Gravity (film)", "Alfonso Cuaron", "scifi", ""),
    ("Moon", 2009, "Moon (2009 film)", "Duncan Jones", "scifi", ""),
    ("Total Recall", 1990, "Total Recall (1990 film)", "Paul Verhoeven", "scifi", ""),
    ("RoboCop", 1987, "RoboCop", "Paul Verhoeven", "scifi", ""),
    ("Terminator 2: Judgment Day", 1991, "Terminator 2: Judgment Day", "James Cameron", "action", ""),
    ("Mad Max: Fury Road", 2015, "Mad Max: Fury Road", "George Miller", "action", ""),
    ("John Wick", 2014, "John Wick", "Chad Stahelski", "action", ""),
    ("Casino Royale", 2006, "Casino Royale (2006 film)", "Martin Campbell", "action", ""),
    ("Skyfall", 2012, "Skyfall", "Sam Mendes", "action", ""),
    ("Mission: Impossible", 1996, "Mission: Impossible (film)", "Brian De Palma", "action", ""),
    ("The Bourne Identity", 2002, "The Bourne Identity (2002 film)", "Doug Liman", "action", ""),
    ("Speed", 1994, "Speed (1994 film)", "Jan de Bont", "action", ""),
    ("Die Hard", 1988, "Die Hard", "John McTiernan", "action", ""),
    ("Predator", 1987, "Predator (film)", "John McTiernan", "action", ""),
    ("Die Hard with a Vengeance", 1995, "Die Hard with a Vengeance", "John McTiernan", "action", ""),
    ("The Hunt for Red October", 1990, "The Hunt for Red October (film)", "John McTiernan", "action", ""),
    # --- Sony/Marvel-adjacent passing-off: Venom-verse and dlist ---
    ("Morbius", 2022, "Morbius (film)", "Daniel Espinosa", "superhero", "passing-off"),
    ("Venom", 2018, "Venom (film)", "Ruben Fleischer", "superhero", "passing-off"),
    ("Madame Web", 2024, "Madame Web (film)", "S. J. Clarkson", "superhero", "passing-off"),
    ("Venom: Let There Be Carnage", 2021, "Venom: Let There Be Carnage", "Andy Serkis", "superhero", "passing-off"),
    # --- drama / awards-bait (a sane catalog has 1000x more of these) ---
    ("The Shawshank Redemption", 1994, "The Shawshank Redemption", "Frank Darabont", "drama", ""),
    ("Forrest Gump", 1994, "Forrest Gump", "Robert Zemeckis", "drama", ""),
    ("Fight Club", 1999, "Fight Club", "David Fincher", "drama", ""),
    ("The Silence of the Lambs", 1991, "The Silence of the Lambs (film)", "Jonathan Demme", "crime", ""),
    ("Schindler's List", 1993, "Schindler's List", "Steven Spielberg", "drama", ""),
    ("Saving Private Ryan", 1998, "Saving Private Ryan", "Steven Spielberg", "war", ""),
    ("Gladiator", 2000, "Gladiator (2000 film)", "Ridley Scott", "historical", ""),
    ("Titanic", 1997, "Titanic (1997 film)", "James Cameron", "drama", ""),
    ("Avatar", 2009, "Avatar (2009 film)", "James Cameron", "scifi", ""),
    ("Avatar: The Way of Water", 2022, "Avatar: The Way of Water", "James Cameron", "scifi", ""),
    ("Aliens", 1986, "Aliens (film)", "James Cameron", "scifi", ""),
    ("The Terminator", 1984, "The Terminator", "James Cameron", "scifi", ""),
    ("Alien", 1979, "Alien (film)", "Ridley Scott", "horror", ""),
    ("The Grand Budapest Hotel", 2014, "The Grand Budapest Hotel", "Wes Anderson", "comedy", ""),
    ("The Big Lebowski", 1998, "The Big Lebowski", "Joel Coen", "comedy", ""),
    ("Fargo", 1996, "Fargo (1996 film)", "Joel Coen", "comedy", ""),
    ("No Country for Old Men", 2007, "No Country for Old Men", "Joel Coen", "crime", ""),
    ("Get Out", 2017, "Get Out", "Jordan Peele", "horror", ""),
    ("Parasite", 2019, "Parasite (2019 film)", "Bong Joon-ho", "drama", ""),
    ("Memories of Murder", 2003, "Memories of Murder", "Bong Joon-ho", "crime", ""),
    ("Train to Busan", 2016, "Train to Busan", "Yeon Sang-ho", "horror", ""),
    ("Spirited Away", 2001, "Spirited Away", "Hayao Miyazaki", "animation", ""),
    ("Princess Mononoke", 1997, "Princess Mononoke", "Hayao Miyazaki", "animation", ""),
    ("Akira", 1988, "Akira (1988 film)", "Katsuhiro Otomo", "animation", ""),
    ("Ghost in the Shell", 1995, "Ghost in the Shell (1995 film)", "Mamoru Oshii", "animation", ""),
    ("Perfect Blue", 1997, "Perfect Blue", "Satoshi Kon", "animation", ""),
    ("Toy Story", 1995, "Toy Story", "John Lasseter", "animation", ""),
    ("Up", 2009, "Up (2009 film)", "Pete Docter", "animation", ""),
    ("WALL-E", 2008, "WALL-E", "Andrew Stanton", "animation", ""),
    ("Everything Everywhere All at Once", 2022, "Everything Everywhere All at Once", "Daniel Kwan", "scifi", ""),
    ("Barbie", 2023, "Barbie (film)", "Greta Gerwig", "comedy", ""),
    ("Oppenheimer's rival", 2024, "Physics (film)", "", "ignore", "SKIP"),
]

# Titles kept in the benchmark but with no reliable source -> filtered at load time.
TARGETS = ["Inception", "The Prestige", "The Dark Knight Rises", "Batman Begins",
           "Joker", "Heat", "The Batman"]

CAST_HINT = re.compile(r"stars? ([A-Z][a-z]+ [A-Z][a-z]+)")
YEAR_HINT = re.compile(r"\b(19|20)\d{2}\b")


def fetch_summary(page: str) -> dict:
    url = API.format(urllib.parse.quote(page.replace(" ", "_"), safe=""))
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as resp:
        return json.loads(resp.read().decode())


def build_entry(name: str, year: int, page: str, director: str,
                franchise: str, note: str) -> dict | None:
    try:
        data = fetch_summary(page)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {name}: {exc}", file=sys.stderr)
        return None
    extract = data.get("extract") or ""
    if not extract:
        return None
    cast: list[str] = []
    match = CAST_HINT.search(extract)
    if match:
        cast.append(match.group(1))
    return {
        "name": data.get("title") or name,
        "year": year,
        "director": director,
        "cast": cast,
        "overview": extract,
        "franchise": franchise,
        "note": note,
    }


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        print(f"{DATA} already exists ({len(json.loads(DATA.read_text('utf-8')))} titles); "
              "delete it to refetch.")
        return
    titles: list[dict] = []
    seen: set[str] = set()
    for name, year, page, director, franchise, note in SEED:
        if note == "SKIP" or name in seen:
            continue
        entry = build_entry(name, year, page, director, franchise, note)
        if entry is None:
            continue
        seen.add(name)
        titles.append(entry)
        print(f"  ok  {entry['name']:<44} {year}  {len(entry['overview']):>4} chars")
        time.sleep(0.15)          # be polite to the Wikimedia API
    DATA.write_text(json.dumps(titles, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(titles)} titles -> {DATA}")


if __name__ == "__main__":
    main()
