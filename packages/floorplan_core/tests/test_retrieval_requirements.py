"""The room descriptor asks for a dining room only when the query does (2026-11-05)."""
from packages.floorplan_core.retrieval import _req_to_room_desc

DINING_SLOT = 4
TOTAL_SLOT = 10


def test_dining_is_off_by_default():
    assert _req_to_room_desc({"bedrooms": 1, "bathrooms": 1})[DINING_SLOT] == 0.0
    assert _req_to_room_desc(None)[DINING_SLOT] == 0.0


def test_dining_on_request_counts_in_the_total():
    off = _req_to_room_desc({"bedrooms": 1, "bathrooms": 1})
    on = _req_to_room_desc({"bedrooms": 1, "bathrooms": 1, "dining": True})
    assert on[DINING_SLOT] == 1.0
    assert on[TOTAL_SLOT] > off[TOTAL_SLOT]
