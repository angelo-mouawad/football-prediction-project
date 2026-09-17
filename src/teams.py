import re

from unidecode import unidecode

# Canonical name, list of spellings seen in the wild
CLUB_ALIASES: dict[str, list[str]] = {
    "Arsenal": ["Arsenal"],
    "Aston Villa": ["Aston Villa"],
    "Birmingham City": ["Birmingham", "Birmingham City"],
    "Blackburn Rovers": ["Blackburn", "Blackburn Rovers"],
    "Blackpool": ["Blackpool"],
    "Bolton Wanderers": ["Bolton", "Bolton Wanderers"],
    "AFC Bournemouth": ["Bournemouth", "AFC Bournemouth"],
    "Brentford": ["Brentford"],
    "Brighton & Hove Albion": [
        "Brighton",
        "Brighton and Hove Albion",
        "Brighton & Hove Albion",
    ],
    "Burnley": ["Burnley"],
    "Cardiff City": ["Cardiff", "Cardiff City"],
    "Charlton Athletic": ["Charlton", "Charlton Ath", "Charlton Athletic"],
    "Coventry City": ["Coventry", "Coventry City"],
    "Chelsea": ["Chelsea"],
    "Crystal Palace": ["Crystal Palace"],
    "Derby County": ["Derby", "Derby County"],
    "Everton": ["Everton"],
    "Fulham": ["Fulham"],
    "Huddersfield Town": ["Huddersfield", "Huddersfield Town"],
    "Hull City": ["Hull", "Hull City"],
    "Ipswich Town": ["Ipswich", "Ipswich Town"],
    "Leeds United": ["Leeds", "Leeds United", "Leeds Utd"],
    "Leicester City": ["Leicester", "Leicester City"],
    "Liverpool": ["Liverpool"],
    "Luton Town": ["Luton", "Luton Town"],
    "Manchester City": ["Man City", "Manchester City"],
    "Manchester United": [
        "Man United",
        "Man Utd",
        "Manchester Utd",
        "Manchester United",
    ],
    "Middlesbrough": ["Middlesbrough", "Middlesboro"],
    "Newcastle United": ["Newcastle", "Newcastle Utd", "Newcastle United"],
    "Norwich City": ["Norwich", "Norwich City"],
    "Nottingham Forest": [
        "Nott'm Forest",
        "Nottm Forest",
        "Nott'ham Forest",
        "Nottingham",
        "Nottingham Forest",
    ],
    "Portsmouth": ["Portsmouth"],
    "Queens Park Rangers": ["QPR", "Queens Park Rangers"],
    "Reading": ["Reading"],
    "Sheffield United": ["Sheffield United", "Sheffield Utd"],
    "Sheffield Wednesday": ["Sheffield Weds", "Sheffield Wednesday"],
    "Southampton": ["Southampton"],
    "Stoke City": ["Stoke", "Stoke City"],
    "Sunderland": ["Sunderland"],
    "Swansea City": ["Swansea", "Swansea City"],
    "Tottenham Hotspur": ["Tottenham", "Tottenham Hotspur", "Spurs"],
    "Watford": ["Watford"],
    "West Bromwich Albion": ["West Brom", "West Bromwich Albion"],
    "West Ham United": ["West Ham", "West Ham United"],
    "Wigan Athletic": ["Wigan", "Wigan Athletic"],
    "Wolverhampton Wanderers": [
        "Wolves",
        "Wolverhampton",
        "Wolverhampton Wanderers",
    ],
}


# Club-type tokens that carry no information. football-data.org returns "Manchester United FC", Understat returns "Manchester United".
NOISE_TOKENS = {"fc", "afc", "cf", "the"}


def _key(name) -> str:
    # Aggressively normalise a name so spelling noise stops mattering.
    if name is None:
        return ""
    s = unidecode(str(name)).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    tokens = [t for t in s.split() if t and t not in NOISE_TOKENS]
    return " ".join(tokens)


_LOOKUP: dict[str, str] = {}
for _canon, _variants in CLUB_ALIASES.items():
    _LOOKUP[_key(_canon)] = _canon
    for _v in _variants:
        _LOOKUP[_key(_v)] = _canon


def to_canonical(name, strict: bool = False):
    # Resolve any spelling to the canonical club name, or None.
    resolved = _LOOKUP.get(_key(name))
    if resolved is None and strict:
        raise KeyError(f"Unmapped club name: {name!r}. Add it to src/teams.py")
    return resolved


def unmapped(names) -> list[str]:
    # Return the sorted unique names that could not be resolved.
    bad = set()
    for n in names:
        if n is None or (isinstance(n, float)):
            continue
        if to_canonical(n) is None:
            bad.add(str(n))
    return sorted(bad)


def canonical_clubs() -> list[str]:
    return sorted(CLUB_ALIASES.keys())
