"""
Generated look-alike data for tests and smoke runs (no real data is needed or used).

It writes the same layout as the challenge dataset:
    <root>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
    <root>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
    <root>/test_labels.tsv          hidden test truth (for local scoring only)

It imitates the noise visible in the real files: all-caps S2 records, repeated tokens,
'(LLC)' / 'L.L.C.' / dropped legal forms, Indic-script names (Devanagari, Tamil, Telugu)
that share no token with the English S1 name, DBA and domain-style names in S3, reordered
names and addresses, zero-padded house numbers ('05131 Copper Meadow Ln'), compound Indian
door numbers, '#5' vs 'Unit 5', missing addresses, chains (one name at many addresses),
about 26% decoy records, France in test only, and test denser than train in S2/S3 per S1.
The absolute difficulty is not calibrated to the real data.

    python tests/synth.py OUT_DIR [--scale 1.0] [--seed 0]
"""
import argparse
import os
import random

from indic_transliteration import sanscript

FIRST = ("james mary robert linda michael sarah david karen joseph nancy thomas lisa daniel betty "
         "mark sandra paul ashley steven kevin brian emily jason olivia ryan grace eric megan").split()
SURN = ("smith johnson williams brown jones garcia miller davis wilson anderson taylor moore "
        "jackson martin lee thompson white harris clark lewis walker hall young allen king wright "
        "olszewski orelee keystone piedmont crestline cobalt altitude jordan").split()
US_BIZ = ("barbershop dental clinic bakery auto repair plumbing roofing consulting realty law "
          "insurance fitness salon cafe grill pharmacy logistics construction landscaping "
          "marketing printing cleaning services solutions group holdings partners retail "
          "supply electric builders media design capital labs studio").split()
US_LEGAL = ["Inc", "LLC", "Corp", "Co", "LLP", "Ltd", "PLLC", ""]
US_STREETS = ("main oak maple cedar pine elm washington lake hill park ridge meadowbrook copper "
              "meadow westchester ellis montebello cameron friendship brigham shoal creek "
              "foxborough cavalier melissa sunset river forest spring valley").split()
US_SUFFIX = [("Street", "St"), ("Avenue", "Ave"), ("Drive", "Dr"), ("Road", "Rd"),
             ("Lane", "Ln"), ("Court", "Ct"), ("Boulevard", "Blvd"), ("Way", "Way")]
US_CITIES = [("Phoenix", "AZ"), ("Dundalk", "MD"), ("High Point", "NC"), ("Tahlequah", "OK"),
             ("Brunswick", "OH"), ("Henrico", "VA"), ("Burlington", "NC"), ("West Jordan", "UT"),
             ("Boston", "MA"), ("Austin", "TX"), ("Denver", "CO"), ("Newark", "NJ"),
             ("Albany", "NY"), ("Tampa", "FL"), ("Salem", "OR"), ("Madison", "WI")]
IN_WORDS = ("shakti shree sai lakshmi ganesh balaji krishna ram om maa durga vision global "
            "sharma gupta patel reddy agarwal kumar singh jain mehta rao iyer nair "
            "agro tech infra food modern real power sun star city royal").split()
IN_BIZ = ("traders enterprises industries systems solutions investment developers medicals "
          "textiles steels foods exports builders motors electricals agencies stores").split()
IN_LEGAL = ["Private Limited", "Pvt Ltd", "Limited", "LLP", ""]
IN_AREAS = ("nerul ambattur kattoor indiranagar koramangala andheri lake town salt lake "
            "banjara hills gachibowli kothrud vashi adyar").split()
IN_CITIES = [("Mumbai", "Maharashtra", "400"), ("Chennai", "Tamil Nadu", "600"),
             ("Kolkata", "West Bengal", "700"), ("Bengaluru", "Karnataka", "560"),
             ("Hyderabad", "Telangana", "500"), ("Pune", "Maharashtra", "411"),
             ("Malappuram", "Kerala", "676"), ("New Delhi", "Delhi", "110")]
FR_WORDS = ("atelier maison boulangerie cabinet garage societe hotel restaurant pharmacie "
            "librairie fleurs jardin lumiere soleil martin bernard dubois moreau laurent").split()
FR_LEGAL = ["SARL", "SAS", "SA", "EURL", ""]
FR_STREETS = ("paix republique victor hugo jean jaures gare eglise moulin pasteur liberte").split()
FR_CITIES = [("Paris", "75"), ("Lyon", "69"), ("Nice", "06"), ("Bourg-en-Bresse", "01"),
             ("Marseille", "13"), ("Lille", "59"), ("Nantes", "44")]
SCRIPTS = [sanscript.DEVANAGARI, sanscript.TAMIL, sanscript.TELUGU]


def _cap(ws):
    return " ".join(w.capitalize() for w in ws)


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.ids = set()

    def uid(self, prefix):
        while True:
            x = f"{prefix}-{self.r.randint(10_000, 999_999_999)}"
            if x not in self.ids:
                self.ids.add(x)
                return x

    # ---- clean S1 entities --------------------------------------------------------------
    def name(self, country):
        r = self.r
        if country == "us":
            core = r.choice([[r.choice(SURN)], [r.choice(FIRST), r.choice(SURN)], [r.choice(SURN)]])
            words = core + r.sample(US_BIZ, r.choice([1, 1, 2]))
            return _cap(words) + (" " + r.choice(US_LEGAL) if r.random() < 0.6 else "")
        if country == "india":
            words = r.sample(IN_WORDS, r.choice([1, 2, 2, 3])) + [r.choice(IN_BIZ)]
            return _cap(words) + (" " + r.choice(IN_LEGAL) if r.random() < 0.7 else "")
        words = r.sample(FR_WORDS, r.choice([1, 2, 2]))
        return _cap(words) + (" " + r.choice(FR_LEGAL) if r.random() < 0.6 else "")

    def address(self, country):
        r = self.r
        if country == "us":
            city, st = r.choice(US_CITIES)
            suf = r.choice(US_SUFFIX)[0]
            a = f"{r.randint(1, 19999)} {_cap([r.choice(US_STREETS)])} {suf}"
            if r.random() < 0.2:
                a += f", Unit {r.randint(1, 400)}"
            a += f", {city}, {st}"
            if r.random() < 0.12:
                a += f" {r.randint(1000, 99999):05d}"
            return a
        if country == "india":
            city, state, pin = r.choice(IN_CITIES)
            door = r.choice([f"{r.randint(1, 99)}", f"{r.randint(1, 9)}-{r.randint(1, 9)}-"
                             f"{r.randint(1, 400)}/{r.randint(1, 99)}", f"{r.randint(1, 9)}/{r.randint(1, 9)}"])
            parts = [f"Shop No {door}" if r.random() < 0.5 else f"Plot No {door}",
                     f"Sector {r.randint(1, 60)}" if r.random() < 0.4 else _cap([r.choice(IN_WORDS)]) + " Road",
                     _cap([r.choice(IN_AREAS)]), city, state]
            a = ", ".join(parts)
            if r.random() < 0.35:
                a += f" {pin}{r.randint(0, 999):03d}"
            return a
        city, dep = r.choice(FR_CITIES)
        return (f"{r.randint(1, 120)} rue de la {r.choice(FR_STREETS).capitalize()}, "
                f"{dep}{r.randint(0, 999):03d} {city}")

    # ---- noisy S2 / S3 copies ------------------------------------------------------------
    def typo(self, s):
        r = self.r
        if len(s) < 4:
            return s
        i = r.randrange(1, len(s) - 1)
        op = r.random()
        if op < 0.4:
            return s[:i] + s[i + 1:]
        if op < 0.7:
            return s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
        return s[:i] + r.choice("aeiourstn") + s[i:]

    def noisy_name(self, name, country, src, dba=None):
        r = self.r
        ws = name.split()
        legal_set = {"Inc", "LLC", "Corp", "Co", "LLP", "Ltd", "PLLC", "Private", "Limited", "Pvt",
                     "SARL", "SAS", "SA", "EURL"}
        body = [w for w in ws if w not in legal_set]
        legal = [w for w in ws if w in legal_set]
        x = r.random()
        if legal and x < 0.15:
            legal = []
        elif legal and x < 0.25:
            legal = ["(" + " ".join(legal) + ")"]
        elif legal == ["LLC"] and x < 0.35:
            legal = ["L.L.C."]
        elif legal == ["Private", "Limited"] and x < 0.4:
            legal = ["Pvt", "Ltd"]
        if len(body) > 1 and r.random() < 0.12:
            r.shuffle(body)
        if r.random() < 0.06:
            j = r.randrange(len(body))
            body.insert(j, body[j])                            # 'Crestline Crestline'
        if r.random() < 0.15:
            j = r.randrange(len(body))
            body[j] = self.typo(body[j])
        out = " ".join(body + legal)
        if country == "india" and r.random() < (0.22 if src == 2 else 0.14):
            itr = out.lower().replace("(", "").replace(")", "").replace(".", "")
            out = sanscript.transliterate(itr, sanscript.ITRANS, r.choice(SCRIPTS))
        elif src == 3 and r.random() < 0.04:
            out = "".join(body).lower() + r.choice([".com", ".net", ".in" if country == "india" else ".org"])
        elif src == 3 and dba and r.random() < 0.4:
            out = out + " DBA " + dba
        elif dba and r.random() < 0.15:
            out = dba                                          # trade name only: no overlap
        if r.random() < (0.28 if src == 2 else 0.08):
            out = out.upper()
        if r.random() < 0.003:
            out = "--" + out
        return out

    def noisy_address(self, addr, country, src):
        r = self.r
        if r.random() < 0.033:
            return ""
        a = addr
        for full, short in US_SUFFIX:
            if r.random() < 0.5:
                a = a.replace(full, short) if full in a else a.replace(short + ",", full + ",")
        if r.random() < 0.2 and ", Unit " in a:
            a = a.replace(", Unit ", " #")
        if country == "us" and r.random() < 0.08:                # zero-padded house number
            a = "0" + a
        segs = [s.strip() for s in a.split(",")]
        if len(segs) > 2 and r.random() < 0.12:
            segs = [segs[-1]] + segs[:-1]                        # 'NC, Canler, 110 ...'
        if len(segs) > 2 and r.random() < 0.15:
            segs.pop(r.randrange(1, len(segs)))                  # an area or city left out
        a = ", ".join(segs)
        if r.random() < 0.12:
            ws = a.split()
            j = r.randrange(len(ws))
            if not ws[j][:1].isdigit():
                ws[j] = self.typo(ws[j])
            a = " ".join(ws)
        if country == "india" and r.random() < 0.05:
            a = sanscript.transliterate(a.lower(), sanscript.ITRANS, r.choice(SCRIPTS))
        if r.random() < (0.66 if src == 2 else 0.03):
            a = a.upper()
        return a


def country_label(r, c):
    if r.random() < 0.01:
        return {"us": "USA", "india": "IND", "france": "FR"}[c]
    return {"us": "US", "india": "India", "france": "France"}[c]


def make_split(g, n_s1, countries, records_per_s1, decoy_share, chain_share):
    """Returns S1 rows, S2 rows, S3 rows and the truth dict S1 id -> list of record ids."""
    r = g.r
    s1, s2, s3, truth = [], [], [], {}
    ents = []
    for i in range(n_s1):
        c = r.choices(list(countries), weights=list(countries.values()))[0]
        if ents and r.random() < chain_share:                    # a branch: same name
            prev = r.choice(ents[-200:])
            if prev["c"] == c:
                name = prev["name"]
            else:
                name = g.name(c)
        else:
            name = g.name(c)
        dba = None
        if r.random() < 0.05:
            dba = g.name(c).rsplit(" ", 1)[0]
        e = {"id": g.uid("S1"), "name": name, "addr": g.address(c), "c": c, "dba": dba}
        ents.append(e)
        s1.append((e["id"], name, e["addr"], country_label(r, c)))
        truth[e["id"]] = []
    n_true = 0
    for e in ents:
        k = 0 if r.random() < 0.056 else max(1, int(r.expovariate(1 / records_per_s1)))
        for _ in range(min(k, 40)):
            src = 2 if r.random() < 0.49 else 3
            rid = g.uid(f"S{src}")
            row = (rid, g.noisy_name(e["name"], e["c"], src, e["dba"]),
                   g.noisy_address(e["addr"], e["c"], src), country_label(r, e["c"]))
            (s2 if src == 2 else s3).append(row)
            truth[e["id"]].append(rid)
            n_true += 1
    n_decoy = int(n_true * decoy_share / (1 - decoy_share))
    for _ in range(n_decoy):
        e = r.choice(ents)
        src = 2 if r.random() < 0.49 else 3
        x = r.random()
        if x < 0.45:                                   # same name, another address
            name, addr = e["name"], g.address(e["c"])
        elif x < 0.7:                                  # similar name, same area
            name = g.name(e["c"])
            addr = e["addr"]
        else:                                          # unrelated business
            name, addr = g.name(e["c"]), g.address(e["c"])
        rid = g.uid(f"S{src}")
        row = (rid, g.noisy_name(name, e["c"], src), g.noisy_address(addr, e["c"], src),
               country_label(r, e["c"]))
        (s2 if src == 2 else s3).append(row)
    r.shuffle(s2)
    r.shuffle(s3)
    return s1, s2, s3, truth


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(x.replace("\t", " ") for x in row) + "\n")


def generate(out, scale=1.0, seed=0):
    g = Gen(seed)
    head = ("entity_id", "business_name", "business_address", "country")
    splits = {
        "train": dict(n_s1=int(12_000 * scale), countries={"us": 0.6, "india": 0.4},
                      records_per_s1=3.4),
        "test": dict(n_s1=int(9_000 * scale), countries={"us": 0.38, "india": 0.47, "france": 0.15},
                     records_per_s1=4.3),
    }
    for split, kw in splits.items():
        d = os.path.join(out, split)
        os.makedirs(d, exist_ok=True)
        s1, s2, s3, truth = make_split(g, decoy_share=0.26, chain_share=0.2, **kw)
        _write(os.path.join(d, f"{split}_source1.tsv"), head, s1)
        _write(os.path.join(d, f"{split}_source2.tsv"), head, s2)
        _write(os.path.join(d, f"{split}_source3.tsv"), head, s3)
        gt = [(k, ",".join(v)) for k, v in truth.items()]
        _write(os.path.join(d, f"{split}_ground_truth.tsv") if split == "train"
               else os.path.join(out, "test_labels.tsv"),
               ("source1_entity_id", "matched_entity_ids"), gt)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate(a.out, a.scale, a.seed)
    print("wrote", a.out)
