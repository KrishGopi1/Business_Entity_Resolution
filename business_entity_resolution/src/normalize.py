"""
Conservative multi-view normalization (v3).

v2 produced one aggressively cleaned name (legal forms, 'india', articles removed) that every
later step had to use. v3 keeps several views of each field, from nearly raw to comparison
only, so that blocking and the model can each use the one that fits and nothing is thrown
away before it has been shown to be noise:

Names
    name_raw_clean     NFKC, case-folded, spacing and decoration cleaned; ORIGINAL script
    name_full          romanized ASCII of the whole name: legal forms (canonical spelling),
                       numbers, the DBA part and token order all kept
    name_core          the name before any DBA marker, minus honorifics and 'M/s' only
    name_base          name_core minus trailing legal forms, a trailing 'india' and a leading
                       'the' (comparison view, what v2 called name_core)
    name_tokens_sorted sorted distinct tokens of name_base without and/of/the/de
    name_nospace       those tokens concatenated ('acme clean' ~ 'acmeclean.com')
    name_phonetic      loose consonant key per token (an extra view, never the only one)
    name_dba           the trade name after 'dba' / 'd/b/a' / 't/a' / 'trading as' (base view)
    legal_form         canonical legal tokens ('pvt ltd', 'llc', 'sarl')
    name_acronym       initials of name_base when it has 2+ tokens
    name_script        'latin', 'devanagari', 'tamil', ... ('none' when empty); flags
                       name_nonlatin, name_is_domain, name_was_upper, name_has_dba
Addresses
    address_raw_clean  NFKC, case-folded, spacing cleaned; original script
    address_full       romanized ASCII tokens, abbreviations and state names canonical, unit
                       designators kept as 'unit' / 'fl' / 'pobox', house numbers without
                       padding zeros, the postcode exactly as written (leading zeros kept),
                       compound door numbers as one token ('8-2-293-82')
    address_tokens     sorted distinct non-stop tokens of address_full
    address_phonetic   phonetic keys of the alphabetic tokens (3+ letters)
    house_number       compound door number, else the number after no/plot/door/h.no, else
                       the first number that is not a unit, floor or postcode
    unit_number        numbers after unit / apt / suite / ste / flat / room / shop / '#'
    floor_number       numbers after 'floor' / 'fl', or before 'floor' ('3rd floor')
    postcode, postcode_prefix (first 3 characters), state (US / Indian code)
    numeric_tokens     sorted distinct numbers other than the postcode (compound numbers
                       also contribute their parts)
    address_missing, address_nonlatin flags

Rules this module keeps (the v3 plan): leading-zero postcodes survive; compound Indian door
numbers survive; floor, unit and house numbers are separate fields; Indic scripts are
transliterated, with the original kept in the raw-clean views and flagged; legal forms stay in
name_full / name_core and in legal_form; generic words ('enterprises', 'services') are never
removed, IDF and the model decide their weight.

Nothing branches on the country label (France has no training data). No external data.
"""
import functools
import multiprocessing as mp
import re
import unicodedata

import pandas as pd
import pyarrow as pa
from indic_transliteration import sanscript

import schema

# ---------------------------------------------------------------------------
# 1. Script handling and shared cleaning
# ---------------------------------------------------------------------------

_SCRIPT_BLOCKS = [  # (first code point, last code point, sanscript scheme, label)
    (0x0900, 0x097F, sanscript.DEVANAGARI, "devanagari"),
    (0x0980, 0x09FF, sanscript.BENGALI, "bengali"),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI, "gurmukhi"),
    (0x0A80, 0x0AFF, sanscript.GUJARATI, "gujarati"),
    (0x0B00, 0x0B7F, sanscript.ORIYA, "oriya"),
    (0x0B80, 0x0BFF, sanscript.TAMIL, "tamil"),
    (0x0C00, 0x0C7F, sanscript.TELUGU, "telugu"),
    (0x0C80, 0x0CFF, sanscript.KANNADA, "kannada"),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM, "malayalam"),
]
_ZW = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_RE_NULL = re.compile(r"<\s*null\s*>", re.I)
_RE_SPACES = re.compile(r"\s+")
_RE_DECOR = re.compile(r"^[\s\-–—_*~.,;:]+|[\s\-–—_*~,;:]+$")   # '--Acme', 'Acme -'
_RE_QUOTES = re.compile(r"[\"“”«»`´]")
_RE_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")
_RE_SCHWA = re.compile(r"(?<=[b-df-hj-np-tv-zBD-HJ-NP-TV-Z])a\b")
_RE_STRAY_DIGIT = re.compile(r"(?<=[A-Za-z])\d+|\d+(?=[A-Za-z])")
_RE_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
_RE_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_RE_LETTERS = re.compile(r"[^a-z ]+")
_LATIN_SPECIAL = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "þ": "th",
                  "ð": "d", "ı": "i", "Æ": "ae", "Œ": "oe", "Ø": "o", "Ł": "l"}
_NAME_MODIFIERS = {"farsi", "dotless", "high", "small", "final", "initial", "medial", "isolated",
                   "capital", "form", "with", "sukun", "hamza", "above", "below"}


def _block_of(ch):
    o = ord(ch)
    if o < 0x0900 or o > 0x0D7F:
        return None
    for lo, hi, scheme, label in _SCRIPT_BLOCKS:
        if lo <= o <= hi:
            return scheme, label
    return None


def detect_script(s):
    """'none' for empty text, the first Indic script found, 'other' for other non-Latin
    letters (Arabic, Cyrillic, CJK ...), else 'latin'."""
    if not s or not s.strip():
        return "none"
    other = False
    for ch in s:
        b = _block_of(ch)
        if b is not None:
            return b[1]
        if ord(ch) > 0x24F and ch.isalpha():
            other = True
    return "other" if other else "latin"


def romanize(s, scheme):
    """One run of one Indic script -> rough ASCII (ITRANS, trailing schwa dropped). Digits
    produced from unmapped signs are removed only when the run had no digit of its own."""
    had_digit = any(unicodedata.category(c) == "Nd" for c in s)
    s = s.replace("ॉ", "ो")                       # Devanagari candra-o ('mॉडर्न' = modern)
    r = sanscript.transliterate(s, scheme, sanscript.ITRANS)
    r = _RE_SCHWA.sub("", r)
    return r if had_digit else _RE_STRAY_DIGIT.sub("", r)


def romanize_runs(s):
    """Transliterate each maximal run of one Indic script separately; ASCII digits and Latin
    text in a mixed string ('प्लॉट 12B') pass through untouched."""
    out, run, run_scheme = [], [], None
    for ch in s:
        b = _block_of(ch)
        sc = b[0] if b else None
        if sc is None and run and not ch.isspace() and unicodedata.combining(ch):
            sc = run_scheme
        if sc is not None and (run_scheme is None or sc == run_scheme):
            run.append(ch)
            run_scheme = sc
            continue
        if run:
            out.append(romanize("".join(run), run_scheme))
            run, run_scheme = [], None
        if sc is not None:
            run, run_scheme = [ch], sc
        else:
            out.append(ch)
    if run:
        out.append(romanize("".join(run), run_scheme))
    return "".join(out)


@functools.lru_cache(maxsize=None)
def _fallback_letter(c):
    """Letters NFKD cannot fold (Arabic, Cyrillic, Greek, 'œ' ...): a rough Latin consonant
    from the Unicode name, enough for the phonetic key. CJK and symbols are dropped."""
    if c in _LATIN_SPECIAL:
        return _LATIN_SPECIAL[c]
    name = unicodedata.name(c, "")
    if " LETTER " not in name:
        return ""
    words = [w.lower() for w in name.split(" LETTER ", 1)[1].split()
             if w.lower() not in _NAME_MODIFIERS]
    word = words[0] if words else ""
    if not word.isalpha():
        return ""
    if name.startswith("LATIN"):
        return word[0]
    m = re.match(r"[^aeiou]*", word).group()
    return m or word[0]


def fold_ascii(s):
    """Accents stripped (é -> e), other letters via _fallback_letter, any script's digits to
    ASCII digits, everything else a space."""
    s = unicodedata.normalize("NFKD", s)
    out = []
    for c in s:
        if ord(c) < 128:
            out.append(c)
        elif unicodedata.combining(c):
            continue
        elif c.isalpha():
            out.append(_fallback_letter(c))
        elif unicodedata.category(c) == "Nd":
            out.append(str(unicodedata.digit(c, "")))
        elif c == "⁄":
            out.append("/")
        else:
            out.append(" ")
    return "".join(out)


def raw_clean(s):
    """NFKC, zero-width and '<null>' removed, case-folded, quotes and leading/trailing
    decoration removed, whitespace collapsed. The original script is kept."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_ZW)
    s = _RE_NULL.sub(" ", s)
    s = _RE_QUOTES.sub("", s).casefold()
    s = _RE_SPACES.sub(" ", s).strip()
    return _RE_DECOR.sub("", s)


def to_ascii(clean):
    """raw_clean text -> (lowercase ASCII text, script label)."""
    script = detect_script(clean)
    s = romanize_runs(clean) if script not in ("latin", "none", "other") else clean
    s = fold_ascii(s).lower()
    s = _RE_DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)   # l.l.c. -> llc
    return s, script


_PHON = [(re.compile(p), r) for p, r in [
    (r"ph", "f"), (r"[bpf]h?", "p"), (r"(ch|jh|j|z)", "j"), (r"(kh|gh|q|c|g|k)", "k"),
    (r"(th|dh|t|d)", "t"), (r"(sh|s|x)", "s"), (r"w", "v")]]
_RE_VOWEL_H = re.compile(r"[aeiouyh]")
_RE_REPEAT = re.compile(r"(.)\1+")


@functools.lru_cache(maxsize=1 << 20)
def phon_key(word):
    """Loose consonant skeleton: 'private', 'praivet', 'prAiveT' -> 'prvt'."""
    w = word.lower()
    if not w:
        return ""
    for pat, rep in _PHON:
        w = pat.sub(rep, w)
    w = w[0] + _RE_VOWEL_H.sub("", w[1:])
    return _RE_REPEAT.sub(r"\1", w)


def dedupe_consecutive(tokens):
    """'crestline crestline clean' -> 'crestline clean'; single letters stay ('c c')."""
    return [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1] or len(t) == 1]


def join_letters(tokens):
    """'j k enterprises' -> 'jk enterprises' (the same as 'J.K.')."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


# ---------------------------------------------------------------------------
# 2. Names
# ---------------------------------------------------------------------------

LEGAL = {  # spelling -> canonical legal tokens
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd", "limited": "ltd", "ltd": "ltd",
    "ltda": "ltd", "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc",
    "pllc": "pllc", "pc": "pc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sa": "sa", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "opc": "opc",
}
LEGAL_VALUES = {t for v in LEGAL.values() for t in v.split()}
LEGAL_PHON = {"prvt": "pvt", "prpt": "pvt", "lmt": "ltd", "lmtt": "ltd"}  # romanized Indic
HONORIFIC = {"sri", "shree", "shri", "sree", "shrii", "srii", "mr", "mrs", "ms", "dr", "smt"}
NAME_STOP = {"and", "of", "the", "de", "du", "des", "et", "d", "l", "la", "le", "les"}
_RE_MS = re.compile(r"^m\s*/\s*s\b\.?")
_RE_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|trading\s+as|doing\s+business\s+as)\b")
_RE_ELISION = re.compile(r"\b([ld])'\s*(?=[a-z])")
_DOMAIN = re.compile(r"^(?:https?://)?www\.|\.(?:com|net|org|io|biz|info|co\.in|in|co|fr|us)\b")
_RE_PAREN_INDIA = re.compile(r"\(\s*india\s*\)")

NAME_FIELDS = ("name_raw_clean", "name_full", "name_core", "name_base", "name_tokens_sorted",
               "name_nospace", "name_phonetic", "name_dba", "legal_form", "name_acronym",
               "name_script", "name_nonlatin", "name_is_domain", "name_was_upper", "name_has_dba")
_NAME_TYPES = (pa.string(),) * 11 + (pa.bool_(),) * 4


def _canon_legal(tokens, nonlatin):
    out = []
    for t in tokens:
        if t in LEGAL:
            out.extend(LEGAL[t].split())
        elif nonlatin and phon_key(t) in LEGAL_PHON:
            out.append(LEGAL_PHON[phon_key(t)])
        else:
            out.append(t)
    return dedupe_consecutive(out)


def _name_tokens(s):
    """ASCII name text -> tokens (legal spellings not yet canonical)."""
    s = _RE_ELISION.sub(r"\1 ", s)                   # l'atelier -> l atelier
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "")
    s = _RE_DIGIT_ALPHA.sub(" ", s)
    s = _RE_NONALNUM.sub(" ", s)
    return join_letters(dedupe_consecutive(s.split()))


def _base(core):
    """name_core tokens -> (name_base tokens, legal tokens). Only the trailing run of legal
    words, a stranded 'and', a trailing 'india' and a leading 'the' leave the base."""
    base, legal = list(core), []
    while base and (base[-1] in LEGAL_VALUES or base[-1] in ("and", "india")):
        t = base.pop()
        if t in LEGAL_VALUES:
            legal.insert(0, t)
    if base and base[0] == "the" and len(base) > 1:
        base.pop(0)
    if not base:                                      # the name is only legal words ('PC')
        base = [t for t in core if t != "and"] or list(core)
        legal = []
    legal = legal or [t for t in core if t in LEGAL_VALUES and t not in base]
    return base, legal


def normalize_name(raw):
    """One raw name -> tuple aligned with NAME_FIELDS."""
    clean = raw_clean(raw)
    s, script = to_ascii(clean)
    nonlatin = script not in ("latin", "none")
    is_domain = bool(_DOMAIN.search(s)) or s.startswith("@")
    s = _DOMAIN.sub(" ", s).replace("@", " ")
    s = _RE_PAREN_INDIA.sub(" india ", s)
    s = _RE_MS.sub(" ", s)
    s = _RE_DBA.sub(" dba ", s)
    tokens = _canon_legal(_name_tokens(s), nonlatin)
    full = list(tokens)
    main, dba = tokens, []
    if "dba" in tokens:
        i = tokens.index("dba")
        main, dba = tokens[:i], tokens[i + 1:]
        if not main:                                  # 'DBA Foo': the trade name is the name
            main, dba = dba, []
    core = [t for t in main if t not in HONORIFIC] or main
    base, legal = _base(core)
    toks = sorted(set(t for t in base if t not in NAME_STOP)) or sorted(set(base))
    ordered = [t for t in base if t not in NAME_STOP] or base
    dba_base = _base([t for t in dba if t not in HONORIFIC] or dba)[0] if dba else []
    acr = "".join(t[0] for t in ordered) if len(ordered) >= 2 else ""
    return (
        clean,
        " ".join(full),
        " ".join(core),
        " ".join(base),
        " ".join(toks),
        "".join(ordered),
        " ".join(phon_key(t) for t in ordered),
        " ".join(dba_base),
        " ".join(legal),
        acr,
        script,
        nonlatin,
        is_domain,
        isinstance(raw, str) and raw.strip() != "" and raw.isupper(),
        bool(dba_base),
    )


# ---------------------------------------------------------------------------
# 3. Addresses
# ---------------------------------------------------------------------------

ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "sainte": "ste", "road": "rd", "avenue": "ave",
    "av": "ave", "drive": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "court": "ct",
    "place": "pl", "highway": "hwy", "parkway": "pkwy", "circle": "cir", "terrace": "ter",
    "square": "sq", "floor": "fl", "flr": "fl", "building": "bldg", "north": "n", "south": "s",
    "east": "e", "west": "w", "route": "rte", "mount": "mt", "chemin": "ch", "impasse": "imp",
    "allee": "all", "near": "nr", "opposite": "opp", "number": "no", "shop": "shp",
    "sector": "sec", "nagar": "ngr", "colony": "col", "post": "po", "office": "off",
    "district": "dist", "dt": "dist", "taluk": "tq", "tal": "tq", "suite": "ste",
    "apartment": "apt", "room": "rm", "hno": "hno", "dno": "dno",
}
UNIT_WORDS = {"unit", "apt", "ste", "flat", "rm", "shp", "off", "pobox"}
FLOOR_WORDS = {"fl"}
HOUSE_MARKERS = {"no", "plot", "door", "hno", "dno", "h"}
ADDR_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "the", "of", "and"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
US_CODES = set(US_STATES.values())
IN_STATES = {  # Telangana / Andhra Pradesh share a code: pairs mix them for one business
    "andhra pradesh": "tgap", "telangana": "tgap", "maharashtra": "mh", "karnataka": "ka",
    "tamil nadu": "tnin", "kerala": "kl", "gujarat": "gj", "rajasthan": "rj", "uttar pradesh": "up",
    "madhya pradesh": "mp", "west bengal": "wb", "bihar": "br", "odisha": "od", "orissa": "od",
    "punjab": "pb", "haryana": "hr", "delhi": "dl", "new delhi": "dl", "assam": "asin",
    "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "himachal pradesh": "hp",
    "goa": "gain", "jammu and kashmir": "jk", "puducherry": "py", "pondicherry": "py",
    "chandigarh": "chin", "tripura": "tr", "meghalaya": "ml", "manipur": "mnin",
    "nagaland": "nl", "mizoram": "mz", "sikkim": "sk", "arunachal pradesh": "arin",
}
_STATE_LOOKUP = {**US_STATES, **IN_STATES, **{v: v for v in US_CODES}}
_RE_US_CODE_ZIP = re.compile(r"\b([a-z]{2})\s+\d{5}(?:\s*-\s*\d{4})?\b")
_RE_ZIP4 = re.compile(r"\b(\d{5})\s*-\s*\d{4}\b")
_RE_POBOX = re.compile(r"\b(?:p\s*o\s*box|post\s*box|bp)\b")
_RE_HNO = re.compile(r"\b([hd])\s*\.?\s*no\b")                    # 'H.No' / 'D.No'
_RE_ORDINAL = re.compile(r"(?<=\d)(st|nd|rd|th)\b")
_RE_HALF = re.compile(r"(?<=\d)\s*1\s*/\s*2\b")
# compound door numbers: '8-2-293/82/a', '3/4', 'g-4', 'c-21', '48/19'
_RE_COMPOUND = re.compile(r"\b(?:[a-z]\s*-\s*)?\d+[a-z]?(?:\s*[-/]\s*(?:\d+[a-z]?|[a-z]\b))+\b"
                          r"|\b[a-z]\s*-\s*\d+[a-z]?\b")
_RE_SPACED_PIN = re.compile(r"\b([1-9]\d{2})\s+(\d{3})\b(?!\s*[-/]?\s*\d)")

ADDR_FIELDS = ("address_raw_clean", "address_full", "address_tokens", "address_phonetic",
               "house_number", "unit_number", "floor_number", "postcode", "postcode_prefix",
               "state", "numeric_tokens", "address_missing", "address_nonlatin")
_ADDR_TYPES = (pa.string(),) * 11 + (pa.bool_(),) * 2


def _zs(t):
    return t.lstrip("0") or "0"


def _find_state(s):
    """State code from a comma segment that is a state name or US code ('dc' wins over a
    'washington' segment), else a US code right before a ZIP ('boston ma 02110')."""
    found = []
    for raw_seg in s.split(","):
        seg = " ".join(_RE_LETTERS.sub(" ", raw_seg).split())
        if seg.split()[-1:] == ["dc"]:
            found.append(("dc", "dc"))
        elif len(seg) == 2 and any(c.isdigit() for c in raw_seg):
            continue                                   # 'Fl 3' is a floor, not Florida
        elif seg in _STATE_LOOKUP:
            found.append((seg, _STATE_LOOKUP[seg]))
        else:                                          # 'west bengal 700001', 'az 85001'
            words = seg.split()
            for n in (3, 2, 1):
                if len(words) >= n and " ".join(words[-n:]) in _STATE_LOOKUP and (
                        n > 1 or len(words[-1]) > 2 or any(c.isdigit() for c in raw_seg)):
                    found.append((" ".join(words[-n:]), _STATE_LOOKUP[" ".join(words[-n:])]))
                    break
    if any(code == "dc" for _, code in found):
        return "dc", []
    if found:
        return found[-1][1], [found[-1][0]]
    code = ""
    for m in _RE_US_CODE_ZIP.finditer(s):
        code = m.group(1)
    return (code, []) if code in US_CODES else ("", [])


def normalize_address(raw):
    """One raw address -> tuple aligned with ADDR_FIELDS."""
    clean = raw_clean(raw)
    s, script = to_ascii(clean)
    missing = s.strip() == ""
    state, names = _find_state(s)
    for w in names:
        if w != state:
            s = re.sub(rf"\b{re.escape(w)}\b", f" {state} ", s)
    s = s.replace("#", " unit ").replace("&", " and ")
    s = _RE_POBOX.sub(" pobox ", s)
    s = _RE_HNO.sub(lambda m: f" {m.group(1)}no ", s)
    s = _RE_ZIP4.sub(r"\1", s)                          # ZIP+4 -> ZIP
    s = _RE_ORDINAL.sub("", s)
    s = _RE_HALF.sub(" ", s)
    if not re.search(r"\b\d{6}\b", s):
        s = _RE_SPACED_PIN.sub(r"\1\2", s)              # '560 038' -> '560038'

    compounds = {}                                      # placeholder -> canonical compound
    def _keep_compound(m):
        parts = re.split(r"\s*[-/]\s*", m.group(0))
        key = "qqcmpd" + "".join(chr(97 + int(c)) for c in str(len(compounds)))
        compounds[key] = "-".join(_zs(p) if p.isdigit() else p for p in parts if p)
        return f" {key} "
    s = _RE_COMPOUND.sub(_keep_compound, s)
    s = _RE_DIGIT_ALPHA.sub(" ", s)
    s = _RE_NONALNUM.sub(" ", s)
    raw_tokens = []
    for t in s.split():
        t = ADDR_ABBR.get(t, t)
        raw_tokens.append(t)

    postcode, pc_pos = "", -1                           # last 5-6 digits, never the 1st token
    for i in range(len(raw_tokens) - 1, 0, -1):
        t = raw_tokens[i]
        if t.isdigit() and len(t) in (5, 6):
            postcode, pc_pos = t, i
            break

    tokens, units, floors, numbers = [], [], [], []
    house_marked, door = "", ""
    for i, t in enumerate(raw_tokens):
        prev = raw_tokens[i - 1] if i else ""
        prev2 = raw_tokens[i - 2] if i > 1 else ""
        nxt = raw_tokens[i + 1] if i + 1 < len(raw_tokens) else ""
        if t in compounds:
            t = compounds[t]
            if prev in UNIT_WORDS or (prev == "no" and prev2 in UNIT_WORDS):
                units.append(t)
            elif not door:
                door = t
            numbers.extend(p for p in t.split("-") if p.isdigit())
            tokens.append(t)
            continue
        if t.isdigit() and i != pc_pos:
            t = _zs(t)
            if prev in FLOOR_WORDS or nxt in FLOOR_WORDS:
                floors.append(t)
            elif prev in UNIT_WORDS or (prev == "no" and prev2 in UNIT_WORDS):
                units.append(t)
            elif prev in HOUSE_MARKERS and not house_marked:
                house_marked = t
            numbers.append(t)
        tokens.append(t)
    tokens = dedupe_consecutive(tokens)
    unit_floor = set(units) | set(floors)
    house = door or house_marked or next(
        (t for t in tokens if t.isdigit() and t != postcode and t not in unit_floor), "")
    alpha = [t for t in tokens if t.isalpha()]
    return (
        clean,
        " ".join(tokens),
        " ".join(sorted(set(t for t in tokens if t not in ADDR_STOP))),
        " ".join(dict.fromkeys(phon_key(t) for t in alpha if len(t) >= 3)),
        house,
        " ".join(sorted(set(units))),
        " ".join(sorted(set(floors))),
        postcode,
        postcode[:3],
        state,
        " ".join(sorted(set(numbers) - {postcode})),
        missing,
        script not in ("latin", "none"),
    )


# ---------------------------------------------------------------------------
# 4. Batch normalization: every distinct string once, in parallel
# ---------------------------------------------------------------------------

def _to_table(rows, fields, types):
    cols = list(zip(*rows)) if rows else [()] * len(fields)
    return pa.table({f: pa.array(list(c), type=t) for f, c, t in zip(fields, cols, types)})


def _names_batch(batch):
    return _to_table([normalize_name(x) for x in batch], NAME_FIELDS, _NAME_TYPES)


def _addrs_batch(batch):
    return _to_table([normalize_address(x) for x in batch], ADDR_FIELDS, _ADDR_TYPES)


def _map(func, items, pool, chunk):
    batches = [items[i:i + chunk] for i in range(0, len(items), chunk)] or [[]]
    parts = pool.map(func, batches, chunksize=1) if pool is not None else [func(b) for b in batches]
    return pa.concat_tables(parts)


def normalize_records(names, addrs, n_jobs=None, chunk=20_000):
    """Equal-length sequences of raw names and addresses -> pyarrow Table with NAME_FIELDS +
    ADDR_FIELDS, one row per record."""
    codes_n, uniq_n = pd.factorize(pd.Series(names, dtype=object).fillna(""), sort=False)
    codes_a, uniq_a = pd.factorize(pd.Series(addrs, dtype=object).fillna(""), sort=False)
    uniq_n, uniq_a = [str(x) for x in uniq_n], [str(x) for x in uniq_a]
    n_jobs = schema.default_jobs(n_jobs)
    big = len(uniq_n) + len(uniq_a) > 4 * chunk
    pool = mp.get_context("spawn").Pool(n_jobs) if n_jobs > 1 and big else None
    try:
        name_tab = _map(_names_batch, uniq_n, pool, chunk)
        addr_tab = _map(_addrs_batch, uniq_a, pool, chunk)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    name_tab = name_tab.take(pa.array(codes_n))
    addr_tab = addr_tab.take(pa.array(codes_a))
    cols = {f: name_tab[f] for f in NAME_FIELDS}
    cols.update({f: addr_tab[f] for f in ADDR_FIELDS})
    return pa.table(cols)
