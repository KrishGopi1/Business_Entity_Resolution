"""
Compact pair features (v3) for the pairwise model.

Every kept candidate pair already carries the candidate ranker's features (blocking.RANK_FEATS:
the three channel cosines with ranks and gaps inside the record, source flags, token overlaps,
postcode / prefix / state / house / unit / name / legal-form agreement, S1-side frequencies,
missingness, script and source flags) plus the ranker's score and rank. This module adds a
small set of string comparisons on the normalized views:

    name    nm_raw_ratio      edit similarity of name_raw_clean (original script)
            nm_full_ratio     edit similarity of name_full (romanized, legal forms kept)
            nm_tsort, nm_tset token-sort / token-set similarity of name_base
            nm_phon_ratio     edit similarity of the phonetic keys
            nm_nospace_jw     Jaro-Winkler of the concatenated name (domain names, spacing)
            nm_exact_full, nm_exact_core
            nm_dba_best       best of record DBA vs S1 name, S1 DBA vs record name, DBA vs DBA
            nm_acronym        one side is the other's acronym
            nm_len_diff
    address ad_tset, ad_tsort token-set / token-sort similarity of address_full
            floor_same
v3.1 (after the real-data loss report: legal words mid-name, domain / handle names, local-script
state names, France department vs region, one-digit door-number decoys):
    name    nl_tset, nl_tsort  token-set / token-sort similarity with legal words and titles
                               removed wherever they sit ('pvt. nuwave dhyechem ltd')
            nl_contain_s, nl_contain_q
                               share of one side's name tokens (3+ letters) found inside the
                               other side's name with spaces removed ('privatesnackrudra.com')
    address ac_tset            token-set similarity of the core address: each side's own state
                               code, France regions / departments and marker words removed,
                               'r' read as 'rue'
            ac_contain_s, ac_contain_q
                               share of one side's core address tokens present in the other
    numbers house_lev, num_lev Levenshtein distance of the door numbers and of the number sets

Left out on purpose (the plan): several partial-ratio variants, counts that grow with the
number of S2/S3 records per S1, source-density features, context passes.
"""
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import distance, fuzz, process

import blocking as bk
import normalize as nz

SRC = ("name_raw_clean", "name_full", "name_core", "name_base", "name_phonetic", "name_nospace",
       "name_dba", "name_acronym", "address_full", "floor_number", "house_number", "numeric_tokens",
       "state")                                        # columns read from the normalized tables
S1_STR = SRC[:-1] + ("name_nolegal", "address_core")    # columns of the Arrow string files
EXTRA = ["nm_raw_ratio", "nm_full_ratio", "nm_tsort", "nm_tset", "nm_phon_ratio", "nm_nospace_jw",
         "nm_exact_full", "nm_exact_core", "nm_dba_best", "nm_acronym", "nm_len_diff",
         "ad_tset", "ad_tsort", "floor_same",
         "nl_tset", "nl_tsort", "nl_contain_s", "nl_contain_q",
         "ac_tset", "ac_contain_s", "ac_contain_q", "house_lev", "num_lev"]
FEATURES = list(bk.RANK_FEATS) + ["rank_score", "rank_pos"] + EXTRA


NAME_DROP = set(nz.LEGAL_VALUES) | {"and", "smt", "shri", "sri", "mr", "mrs", "ms", "dr", "messrs"}
ADDR_MARKERS = {"unit", "no", "n"}
FR_MULTI = """auvergne rhone alpes|bourgogne franche comte|centre val de loire|grand est|hauts de france|
ile de france|nouvelle aquitaine|pays de la loire|provence alpes cote d azur|basse normandie|
haute normandie|champagne ardenne|franche comte|languedoc roussillon|midi pyrenees|
nord pas de calais|poitou charentes|rhone alpes|alpes de haute provence|hautes alpes|
alpes maritimes|bouches du rhone|charente maritime|corse du sud|haute corse|cote d or|
cotes d armor|eure et loir|haute garonne|ille et vilaine|indre et loire|loir et cher|haute loire|
loire atlantique|lot et garonne|maine et loire|haute marne|meurthe et moselle|puy de dome|
pyrenees atlantiques|hautes pyrenees|pyrenees orientales|bas rhin|haut rhin|haute saone|
saone et loire|haute savoie|seine maritime|seine et marne|deux sevres|tarn et garonne|haute vienne|
territoire de belfort|hauts de seine|seine st denis|val de marne|val d oise|pas de calais"""
FR_SINGLE = set("""bretagne corse normandie occitanie alsace aquitaine auvergne bourgogne limousin
lorraine picardie paca ain aisne allier ardeche ardennes ariege aube aude aveyron calvados cantal
charente cher correze creuse dordogne doubs drome eure finistere gard gers gironde herault indre
isere jura landes loire loiret lot lozere manche marne mayenne meuse morbihan moselle nievre nord
oise orne rhone sarthe savoie somme tarn var vaucluse vendee vienne vosges yonne essonne yvelines
france""".split())
_RE_FR = re.compile(r"\b(?:" + "|".join(x.strip() for x in FR_MULTI.split("|")) + r")\b")


def name_nolegal(name_full):
    return " ".join(t for t in name_full.split() if t not in NAME_DROP)


def address_core(address_full, state):
    """Street, number and city tokens: the row's own state code, France regions and
    departments (phrases anywhere, single words at the end) and marker words removed."""
    s = _RE_FR.sub(" ", address_full)
    toks = ["rue" if t == "r" else t for t in s.split() if t not in ADDR_MARKERS and t != state]
    while toks and toks[-1] in FR_SINGLE:
        toks.pop()
    return " ".join(toks)


def write_strings(tab, path):
    """Saves the string columns the features need as an uncompressed Arrow file, which every
    worker memory-maps (one copy in the page cache instead of Python strings per process).
    tab holds the SRC columns; the derived views are computed here once per row."""
    col = {c: pc.fill_null(tab[c], "") for c in SRC}
    nf, af, st = (col[c].to_pylist() for c in ("name_full", "address_full", "state"))
    col["name_nolegal"] = pa.array([name_nolegal(x) for x in nf], pa.string())
    col["address_core"] = pa.array([address_core(a, s) for a, s in zip(af, st)], pa.string())
    t = pa.table({c: col[c] for c in S1_STR})
    with pa.OSFile(path, "wb") as f, pa.ipc.new_file(f, t.schema) as w:
        w.write_table(t)


def open_strings(path):
    return pa.ipc.open_file(pa.memory_map(path, "r")).read_all()


def _take(tab, rows):
    idx = pa.array(np.asarray(rows, np.int64))
    return {c: tab.column(c).take(idx).to_numpy(zero_copy_only=False) for c in S1_STR}


def _sim(a, b, scorer, workers, scale=100.0):
    out = np.asarray(process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32),
                     dtype=np.float32) / scale
    out[(a == "") | (b == "")] = np.nan
    return out


def _eq(a, b):
    out = (a == b).astype(np.float32)
    out[(a == "") | (b == "")] = np.nan
    return out


def _contain(a, b, min_len=3):
    """Share of the tokens of a (min_len+ characters) found inside b with spaces removed."""
    out = np.full(len(a), np.nan, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        toks = [t for t in x.split() if len(t) >= min_len]
        if toks and y:
            y = y.replace(" ", "")
            out[i] = sum(t in y for t in toks) / len(toks)
    return out


def _tok_share(a, b):
    """Share of the tokens of a present among the tokens of b."""
    out = np.full(len(a), np.nan, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        tx = x.split()
        if tx and y:
            ty = set(y.split())
            out[i] = sum(t in ty for t in tx) / len(tx)
    return out


def pair_features(pairs, S, Q, workers=1):
    """pairs: dict of arrays from a candidate chunk (qi, si, the ranker features, rank_score,
    rank_pos). S, Q: open_strings() tables of the partition's S1 and records.
    Returns a float32 matrix in FEATURES order (NaN = missing)."""
    qi, si = np.asarray(pairs["qi"]), np.asarray(pairs["si"])
    n = len(qi)
    if n == 0:
        return np.empty((0, len(FEATURES)), np.float32)
    s, q = _take(S, si), _take(Q, qi)
    F = {c: np.asarray(pairs[c], np.float32) for c in list(bk.RANK_FEATS) + ["rank_score", "rank_pos"]}
    F["nm_raw_ratio"] = _sim(s["name_raw_clean"], q["name_raw_clean"], fuzz.ratio, workers)
    F["nm_full_ratio"] = _sim(s["name_full"], q["name_full"], fuzz.ratio, workers)
    F["nm_tsort"] = _sim(s["name_base"], q["name_base"], fuzz.token_sort_ratio, workers)
    F["nm_tset"] = _sim(s["name_base"], q["name_base"], fuzz.token_set_ratio, workers)
    F["nm_phon_ratio"] = _sim(s["name_phonetic"], q["name_phonetic"], fuzz.ratio, workers)
    F["nm_nospace_jw"] = _sim(s["name_nospace"], q["name_nospace"],
                              distance.JaroWinkler.normalized_similarity, workers, 1.0)
    F["nm_exact_full"] = _eq(s["name_full"], q["name_full"])
    F["nm_exact_core"] = _eq(s["name_core"], q["name_core"])
    dba = [_sim(s["name_base"], q["name_dba"], fuzz.token_set_ratio, workers)]
    if (s["name_dba"] != "").any():
        dba += [_sim(s["name_dba"], q["name_base"], fuzz.token_set_ratio, workers),
                _sim(s["name_dba"], q["name_dba"], fuzz.token_set_ratio, workers)]
    F["nm_dba_best"] = np.fmax.reduce(dba) if len(dba) > 1 else dba[0]
    F["nm_acronym"] = (((s["name_acronym"] != "") & (q["name_nospace"] == s["name_acronym"]))
                       | ((q["name_acronym"] != "") & (s["name_nospace"] == q["name_acronym"]))
                       ).astype(np.float32)
    ls = np.fromiter((len(x) for x in s["name_base"]), np.float32, n)
    lq = np.fromiter((len(x) for x in q["name_base"]), np.float32, n)
    F["nm_len_diff"] = np.abs(ls - lq)
    F["ad_tset"] = _sim(s["address_full"], q["address_full"], fuzz.token_set_ratio, workers)
    F["ad_tsort"] = _sim(s["address_full"], q["address_full"], fuzz.token_sort_ratio, workers)
    F["floor_same"] = _eq(s["floor_number"], q["floor_number"])
    F["nl_tset"] = _sim(s["name_nolegal"], q["name_nolegal"], fuzz.token_set_ratio, workers)
    F["nl_tsort"] = _sim(s["name_nolegal"], q["name_nolegal"], fuzz.token_sort_ratio, workers)
    F["nl_contain_s"] = _contain(s["name_nolegal"], q["name_nolegal"])
    F["nl_contain_q"] = _contain(q["name_nolegal"], s["name_nolegal"])
    F["ac_tset"] = _sim(s["address_core"], q["address_core"], fuzz.token_set_ratio, workers)
    F["ac_contain_s"] = _tok_share(s["address_core"], q["address_core"])
    F["ac_contain_q"] = _tok_share(q["address_core"], s["address_core"])
    F["house_lev"] = _sim(s["house_number"], q["house_number"], distance.Levenshtein.distance, workers, 1.0)
    F["num_lev"] = _sim(s["numeric_tokens"], q["numeric_tokens"], distance.Levenshtein.distance, workers, 1.0)
    return np.column_stack([F[c] for c in FEATURES]).astype(np.float32)
