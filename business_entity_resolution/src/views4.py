"""
v4 extra views, derived from v3's normalized tables (work/<split>/norm). v3's normalize.py is
not changed, so every cached v3 stage (prepare, contexts, candidates, ranker) stays valid; the
fixes below live only in these new columns.

Names
    name_nolegal   name_full with legal words and titles removed wherever they sit
                   ('pvt. nuwave dhyechem ltd' -> 'nuwave dhyechem'; v3 only strips a trailing run)
    name_loose     a loose romanization of name_nolegal on which ITRANS output and everyday
                   English spellings of an Indian name converge: digraphs folded (sh -> s,
                   ph -> f, th -> t ...), c/q/x/z/w mapped, every vowel read as 'a', repeated
                   letters collapsed, a final vowel dropped ('shrI lakShmI', 'shree laxmi' ->
                   'sr laksm'; 'mIDiyA' and 'media' -> 'mad')
    name_despaced  name_nolegal without spaces, for domains and handles
                   ('privatesnackrudra.com' vs 'rudra snack pvt ltd')
    loose_key      sorted distinct name_loose tokens (an exact block key)
Addresses
    address_core   address_full without the record's own state (also when it was written in an
                   Indic script: 'महाराष्ट्र' is 'maharashtr' after v3's romanization and was not
                   recognized), without France regions / departments and country words, marker
                   words ('unit', 'no') dropped, 'r' read as 'rue'
    state4         v3's state code, or the one found in an Indic-script segment
    numbers4       every number of the address with padding zeros removed, compound door numbers
                   also split into their parts (sorted, distinct; the postcode excluded)
"""
import functools
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import features as ft3
import normalize as nz

V4_FIELDS = ("name_nolegal", "name_loose", "name_despaced", "loose_key", "address_core", "state4",
             "numbers4", "q_is_handle")
NORM_COLS = ("name_raw_clean", "name_full", "name_is_domain", "name_nonlatin", "address_raw_clean",
             "address_full", "state", "postcode", "numeric_tokens", "house_number")

NAME_DROP = ft3.NAME_DROP | nz.HONORIFIC
_DIGRAPHS = [("ksh", "ks"), ("sch", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"),
             ("kh", "k"), ("jh", "j"), ("ph", "f"), ("sh", "s"), ("zh", "j"), ("ch", "c"),
             ("ck", "k"), ("wh", "v")]
_RE_C_SOFT = re.compile(r"c(?=[eiy])")
_RE_G_SOFT = re.compile(r"(?<=.)g(?=[eiy])")        # technology / teknoloji
_TRANS = str.maketrans({"q": "k", "z": "j", "w": "v", "y": "a",
                        "e": "a", "i": "a", "o": "a", "u": "a"})
_RE_REP = re.compile(r"(.)\1+")
_RE_M_NASAL = re.compile(r"m(?=[^aeiouybpmv\W\d])")


@functools.lru_cache(maxsize=1 << 20)
def loose_token(t):
    """Loose romanization of one ASCII token (see the module docstring). Digits pass."""
    if not t or not t.isalpha():
        return t
    w = t.lower()
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    w = _RE_C_SOFT.sub("s", w)
    w = _RE_G_SOFT.sub("j", w)
    w = w.replace("c", "k").replace("x", "ks")
    w = _RE_M_NASAL.sub("n", w)                     # anusvara read as m: 'telamgana'
    w = w[0] + w[1:].translate(_TRANS) if w[0] == "y" else w.translate(_TRANS)
    w = _RE_REP.sub(r"\1", w)
    if len(w) > 2 and w[-1] == "a":
        w = w[:-1]
    return w


# Legal words as ITRANS spells them from Indic script ('एलएलपी' -> 'elaelapi', 'எல்எல்பி' ->
# 'elelbhi', 'प्रा. लि.' -> 'pra li'), seen in the real loss sample.
_INDIC_LEGAL = {"elelpi", "elelbhi", "elaelapi", "elaelpi", "ellelpi", "elelapi", "limitedd",
                "praivet", "privet", "praivhet", "limited", "limitied", "limited."}


def name_nolegal(name_full):
    toks = [t for t in name_full.split() if t not in NAME_DROP and t not in _INDIC_LEGAL]
    for i in range(len(toks) - 1):                  # 'pra li' = pvt ltd
        if toks[i] == "pra" and toks[i + 1] == "li":
            toks = toks[:i] + toks[i + 2:]
            break
    return " ".join(toks)


def skeleton(text):
    """Consonant skeleton of a loose romanization: first letter kept, then vowels removed."""
    w = "".join(loose_token(t) for t in text.split() if t.isalpha())
    return (w[:1] + re.sub(r"[a]", "", w[1:])) if w else ""


# Indian states and union territories, with the native-language names that v3's romanization
# of the local script produces ('pashchim banga', 'keralam', 'dilli' ...)
_IN_STATE_NAMES = {
    **{k: v for k, v in nz.IN_STATES.items()},
    "pashchim banga": "wb", "pashchimbanga": "wb", "pashchim bangal": "wb", "paschimbanga": "wb",
    "keralam": "kl", "tamizhnadu": "tnin", "tamilnadu": "tnin", "tamizh nadu": "tnin",
    "dilli": "dl", "hariyana": "hr", "odisa": "od", "uttarpradesh": "up", "madhyapradesh": "mp",
    "andhrapradesh": "tgap", "andhra": "tgap", "telengana": "tgap", "karnatak": "ka",
    "gujrat": "gj", "panjab": "pb", "jharkhand": "jh", "chhatisgarh": "cg", "uttaranchal": "uk",
    "maharashtra": "mh", "mumbai": "", "bharat": "", "bharata": "",
}
_STATE_SKEL = {}
for _n, _c in _IN_STATE_NAMES.items():
    _sk = skeleton(_n)
    if _c and len(_sk) >= 3:
        _STATE_SKEL.setdefault(_sk, _c)
_COUNTRY_WORDS = {"india", "bharat", "usa", "us", "united", "states", "america", "france",
                  "republic", "french", "francaise"}


def _lev1(a, b):
    """True if a and b differ by at most one insertion, deletion or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:] or (len(a) == len(b) and a[i + 1:] == b[i + 1:])


def indic_state(address_raw_clean):
    """(state code, romanized tokens of that segment) for the first comma segment written in an
    Indic script whose loose skeleton matches a state name ('', []) otherwise."""
    if not address_raw_clean or all(ord(c) < 0x0900 or ord(c) > 0x0D7F for c in address_raw_clean):
        return "", []
    for seg in address_raw_clean.split(","):
        if all(ord(c) < 0x0900 or ord(c) > 0x0D7F for c in seg):
            continue
        asc, _ = nz.to_ascii(seg.strip())
        toks = re.sub(r"[^a-z ]+", " ", asc).split()
        if not toks:
            continue
        sk = skeleton(" ".join(toks))
        code = _STATE_SKEL.get(sk)
        if code is None and len(sk) >= 5:
            code = next((c for k, c in _STATE_SKEL.items() if len(k) >= 5 and _lev1(k, sk)), None)
        if code:
            return code, toks
    return "", []


def address_core(address_full, state, address_raw_clean):
    """v3.1's core address (own state code, France regions / departments, markers removed,
    'r' as 'rue') plus the Indic-script state segment and country words removed."""
    st4, drop = indic_state(address_raw_clean)
    core = ft3.address_core(address_full, state)
    toks = core.split()
    if drop:
        dset = set(drop)
        toks = [t for t in toks if t not in dset]
    toks = [t for t in toks if t not in _COUNTRY_WORDS and t != st4]
    return " ".join(toks), (state or st4)


def numbers4(address_full, postcode):
    out = set()
    for t in address_full.split():
        parts = t.split("-") if "-" in t else [t]
        for p in parts + ([t] if len(parts) > 1 else []):
            if any(c.isdigit() for c in p):
                p = p.lstrip("0") or "0" if p.isdigit() else p
                if p != postcode:
                    out.add(p)
    return " ".join(sorted(out))


def compute_views(tab):
    """tab: a v3 norm table with NORM_COLS -> pyarrow table with V4_FIELDS (one row per row).
    Distinct (name, address) values are computed once."""
    cols = {c: pc.fill_null(tab[c], "" if tab.schema.field(c).type == pa.string() else False)
            .to_pylist() for c in ("name_full", "name_is_domain", "address_full", "state",
                                    "address_raw_clean", "postcode")}
    names = {}
    addrs = {}
    out = {f: [] for f in V4_FIELDS}
    for nf, dom, af, st, ar, pcd in zip(cols["name_full"], cols["name_is_domain"], cols["address_full"],
                                        cols["state"], cols["address_raw_clean"], cols["postcode"]):
        nk = (nf, dom)
        if nk not in names:
            nl = name_nolegal(nf)
            lt = [x for x in (loose_token(t) for t in nl.split()) if x]
            names[nk] = (nl, " ".join(lt), nl.replace(" ", ""), " ".join(sorted(set(lt))),
                         bool(dom) or (len(nl.split()) == 1 and len(nl) >= 8))
        ak = (af, st, ar, pcd)
        if ak not in addrs:
            core, st4 = address_core(af, st, ar)
            addrs[ak] = (core, st4, numbers4(af, pcd))
        n, a = names[nk], addrs[ak]
        out["name_nolegal"].append(n[0])
        out["name_loose"].append(n[1])
        out["name_despaced"].append(n[2])
        out["loose_key"].append(n[3])
        out["q_is_handle"].append(n[4])
        out["address_core"].append(a[0])
        out["state4"].append(a[1])
        out["numbers4"].append(a[2])
    types = {f: (pa.bool_() if f == "q_is_handle" else pa.string()) for f in V4_FIELDS}
    return pa.table({f: pa.array(v, type=types[f]) for f, v in out.items()})


def _views_chunk(tab):
    return compute_views(tab)


def compute_views_parallel(tab, n_jobs=1, chunk=200_000):
    n = tab.num_rows
    if n_jobs <= 1 or n <= 2 * chunk:
        return compute_views(tab)
    import multiprocessing as mp
    parts = [tab.slice(i, chunk) for i in range(0, n, chunk)]
    with mp.get_context("spawn").Pool(n_jobs) as pool:
        res = pool.map(_views_chunk, parts, chunksize=1)
    return pa.concat_tables(res)
