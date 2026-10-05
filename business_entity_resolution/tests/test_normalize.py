"""Conservative normalization: the rules of the v3 plan."""
import normalize as nz

N = dict(zip(nz.NAME_FIELDS, range(len(nz.NAME_FIELDS))))
A = dict(zip(nz.ADDR_FIELDS, range(len(nz.ADDR_FIELDS))))


def name(raw, field):
    return nz.normalize_name(raw)[N[field]]


def addr(raw, field):
    return nz.normalize_address(raw)[A[field]]


def test_leading_zero_postcodes_survive():
    assert addr("12 rue de la Paix, 06000 Nice", "postcode") == "06000"
    assert addr("100 Main St, Boston MA 02110-1234", "postcode") == "02110"
    assert addr("100 Main St, Boston MA 02110-1234", "state") == "ma"
    assert addr("12 rue de la Paix, 06000 Nice", "postcode_prefix") == "060"


def test_padded_house_numbers_are_unpadded_but_postcode_is_not():
    assert addr("05131 Copper Meadow Ln, West Jordan", "house_number") == "5131"
    assert addr("H.NO 008 GOVIND BAUG", "house_number") == "8"


def test_compound_indian_door_numbers_survive():
    assert addr("8-2-293/82, Road No 12, Banjara Hills, Hyderabad", "house_number") == "8-2-293-82"
    assert addr("Plot No 3/4, MG Road", "house_number") == "3-4"
    assert "8-2-293-82" in addr("8-2-293/82, Road No 12", "address_full").split()


def test_floor_unit_and_house_are_separate():
    a = "3rd Floor, Unit 5, 100 Main St"
    assert addr(a, "floor_number") == "3"
    assert addr(a, "unit_number") == "5"
    assert addr(a, "house_number") == "100"
    assert addr("100 Main St #5", "unit_number") == addr("Unit 5, 100 Main St", "unit_number") == "5"
    assert addr("SHP NO. G-4, PLOT NO. 19, SECTOR 42", "unit_number") == "g-4"
    assert addr("SHP NO. G-4, PLOT NO. 19, SECTOR 42", "house_number") == "19"


def test_legal_forms_kept_in_full_view_and_as_evidence():
    assert name("Olszewski Holding Company LLC LLC", "name_full") == "olszewski holding co llc"
    assert name("Olszewski Holding Company LLC LLC", "name_base") == "olszewski holding"
    assert name("Olszewski Holding Company LLC LLC", "legal_form") == "co llc"
    assert name("Piedmont Complete Chile L.L.C.", "legal_form") == "llc"
    assert name("COBALT (LLC)", "name_full") == "cobalt llc"


def test_generic_words_are_not_removed():
    assert name("Sai Enterprises", "name_base") == "sai enterprises"
    assert name("Global Services Group", "name_tokens_sorted") == "global group services"


def test_honorifics_only_leave_core():
    assert name("M/s Shree Vision Systems Pvt Ltd", "name_full") == "shree vision systems pvt ltd"
    assert name("M/s Shree Vision Systems Pvt Ltd", "name_core") == "vision systems pvt ltd"


def test_dba_kept_separately_and_in_full_view():
    raw = "John Smith DBA Smith Plumbing"
    assert name(raw, "name_dba") == "smith plumbing"
    assert name(raw, "name_core") == "john smith"
    assert "plumbing" in name(raw, "name_full")


def test_indic_transliterated_with_original_script_kept():
    raw = "रियल मॉडर्न फूड लिमिटेड"
    r = nz.normalize_name(raw)
    assert r[N["name_script"]] == "devanagari" and r[N["name_nonlatin"]]
    assert r[N["name_raw_clean"]] == raw                       # original script retained
    assert r[N["legal_form"]] == "ltd"
    assert r[N["name_phonetic"]] == nz.normalize_name("Real Modern Food Ltd")[N["name_phonetic"]]


def test_token_order_kept_in_full_sorted_in_comparison_view():
    assert name("Keystone Odyssey", "name_full") == "keystone odyssey"
    assert name("Odyssey Keystone", "name_tokens_sorted") == name("Keystone Odyssey", "name_tokens_sorted")


def test_empty_inputs():
    assert name("", "name_full") == "" and name(None, "name_script") == "none"
    assert addr("", "address_missing") is True
