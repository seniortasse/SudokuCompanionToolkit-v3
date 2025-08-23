# tests/test_payload_schema.py
# This test only checks shape; adapt later to run the real CLI if desired.
def test_payload_shape_example():
    payload = {"rectify": {"warped":"x.png","cells_json":"y.json"},
               "moves": [], "overlays": []}
    assert set(payload.keys()) == {"rectify","moves","overlays"}
    assert "warped" in payload["rectify"]
    assert "cells_json" in payload["rectify"]