from khoya.query import parse_query

STATIONS = ["Borivali", "Mumbai Central", "Mumbai CSMT", "Pune Junction", "Dadar"]


def test_extracts_train_station_and_description():
    q = parse_query("I lost my black Wildcraft backpack on 12951 near Borivali", STATIONS)
    assert q.train_no == "12951"
    assert q.station == "Borivali"
    assert q.clip_text == "a photo of a black wildcraft backpack"


def test_station_alias_and_train_prefix():
    q = parse_query("left my umbrella at CST, train no. 12123", STATIONS)
    assert q.station == "Mumbai CSMT"
    assert q.train_no == "12123"
    assert q.clip_text == "a photo of an umbrella"


def test_keeps_descriptive_words():
    q = parse_query("blue suitcase with wheels and a broken handle", STATIONS)
    assert q.clip_text == "a photo of a blue suitcase with wheels and a broken handle"
    assert q.train_no is None and q.station is None


def test_only_logistics_gives_no_visual_query():
    q = parse_query("lost on 12951 at Dadar", STATIONS)
    assert q.clip_text is None
    assert q.train_no == "12951" and q.station == "Dadar"


def test_empty():
    assert parse_query("   ", STATIONS).clip_text is None
    assert parse_query(None, STATIONS).train_no is None
