"""Canadian FWI System equation tests (Phase 2) — property-based sanity."""

from __future__ import annotations

from wildfire_marl.data.fwi import FWIState, daily_fwi


def _run_days(n, temp, rh, wind, rain=0.0, month=6):
    state = FWIState()
    codes = None
    for _ in range(n):
        state, codes = daily_fwi(temp=temp, rh=rh, wind=wind, rain=rain, month=month, prev=state)
    return codes


def test_hot_dry_windy_beats_cool_humid_calm():
    hot = _run_days(10, temp=35.0, rh=20.0, wind=20.0)
    cool = _run_days(10, temp=15.0, rh=70.0, wind=5.0)
    for k in ("ffmc", "isi", "bui", "fwi"):
        assert hot[k] > cool[k], f"{k}: hot/dry should exceed cool/humid"


def test_drought_codes_accumulate_without_rain():
    d3 = _run_days(3, temp=30.0, rh=25.0, wind=10.0)
    d20 = _run_days(20, temp=30.0, rh=25.0, wind=10.0)
    assert d20["dmc"] > d3["dmc"]
    assert d20["dc"] > d3["dc"]


def test_rain_reduces_ffmc():
    state = FWIState()
    for _ in range(5):
        state, dry = daily_fwi(temp=30.0, rh=25.0, wind=10.0, rain=0.0, month=6, prev=state)
    _, wet = daily_fwi(temp=30.0, rh=25.0, wind=10.0, rain=15.0, month=6, prev=state)
    assert wet["ffmc"] < dry["ffmc"]


def test_outputs_in_plausible_operational_ranges():
    c = _run_days(30, temp=35.0, rh=20.0, wind=15.0)  # extreme June desert
    assert 80.0 < c["ffmc"] <= 101.0
    assert c["fwi"] > 10.0  # high fire danger
    calm = _run_days(2, temp=10.0, rh=90.0, wind=2.0)
    assert calm["fwi"] < 5.0  # low fire danger


def test_deterministic():
    assert _run_days(7, 25.0, 40.0, 12.0) == _run_days(7, 25.0, 40.0, 12.0)
