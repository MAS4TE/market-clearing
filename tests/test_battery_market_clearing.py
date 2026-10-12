import base64
import importlib
import json
import sys
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace

import pyomo.environ as pyo
import pytest


def _install_assume_stubs():
    assume = ModuleType("assume")
    common = ModuleType("assume.common")
    market_objects = ModuleType("assume.common.market_objects")
    markets = ModuleType("assume.markets")
    base_market = ModuleType("assume.markets.base_market")

    market_objects.MarketConfig = dict
    market_objects.Order = dict
    market_objects.Orderbook = list

    class MarketRole:
        def __init__(self, marketconfig):
            self.marketconfig = marketconfig

        def validate_orderbook(self, orderbook, agent_addr):
            del orderbook, agent_addr

    base_market.MarketRole = MarketRole

    sys.modules.setdefault("assume", assume)
    sys.modules.setdefault("assume.common", common)
    sys.modules.setdefault("assume.common.market_objects", market_objects)
    sys.modules.setdefault("assume.markets", markets)
    sys.modules.setdefault("assume.markets.base_market", base_market)


_install_assume_stubs()
module = importlib.import_module("mas4te_market_clearing.battery_market_clearing")
BatteryClearing = module.BatteryClearing
calculate_meta = module.calculate_meta


def _get_available_solver():
    # Prefer open/local LP solvers first. Gurobi can appear "available" but fail at
    # runtime due to missing/blocked license token access in sandboxed runs.
    for solver_name in ("highs", "appsi_highs", "glpk", "cbc", "gurobi"):
        try:
            if pyo.SolverFactory(solver_name).available(False):
                return solver_name
        except Exception:
            continue
    return None


SOLVER_NAME = _get_available_solver()


class IntegrationBatteryClearing(BatteryClearing):
    def solve_model(self, model: pyo.ConcreteModel, solver: str = "highs") -> None:
        if SOLVER_NAME is None:
            pytest.skip(
                "No supported LP solver available (highs/appsi_highs/glpk/cbc)."
            )
        super().solve_model(model, solver=SOLVER_NAME)


def _make_role(
    allowed_c_rates=(0.5, 1.0),
    *,
    locations=None,
    exclusive_link_field="exclusive_id",
):
    param_dict = {"allowed_c_rates": list(allowed_c_rates)}
    if locations is not None:
        param_dict["locations"] = list(locations)
    if exclusive_link_field is not None:
        param_dict["exclusive_link_field"] = exclusive_link_field
    marketconfig = SimpleNamespace(param_dict=param_dict)
    return IntegrationBatteryClearing(marketconfig)


def test_calculate_meta_with_weighted_average():
    start = datetime(2026, 1, 1, 10, 0)
    end = start + timedelta(hours=2)
    product = (start, end, None)
    accepted_supply_orders = [
        {"accepted_volume": 2, "accepted_price": 10},
        {"accepted_volume": 4, "accepted_price": 16},
    ]
    accepted_demand_orders = [
        {"accepted_volume": -3, "accepted_price": 20},
        {"accepted_volume": -3, "accepted_price": 22},
    ]

    meta = calculate_meta(accepted_supply_orders, accepted_demand_orders, product)

    assert meta["supply_volume"] == 6
    assert meta["demand_volume"] == 6
    assert meta["supply_volume_energy"] == 12
    assert meta["demand_volume_energy"] == 12
    assert meta["price"] == pytest.approx((2 * 10 + 4 * 16) / 6)
    assert meta["min_price"] == 10
    assert meta["max_price"] == 16


def test_calculate_meta_without_supply_orders_defaults_to_zero_price():
    start = datetime(2026, 1, 1, 10, 0)
    end = start + timedelta(hours=1)
    product = (start, end, None)

    meta = calculate_meta([], [], product)

    assert meta["supply_volume"] == 0
    assert meta["demand_volume"] == 0
    assert meta["price"] == 0
    assert meta["min_price"] == 0
    assert meta["max_price"] == 0


def test_validate_orderbook_rejects_disallowed_c_rate():
    role = _make_role(allowed_c_rates=(0.5, 1.0))

    with pytest.raises(ValueError, match="is not in"):
        role.validate_orderbook([{"c_rate": 2.0}], agent_addr="agent")


def test_validate_orderbook_accepts_allowed_c_rates():
    role = _make_role(allowed_c_rates=(0.5, 1.0))
    role.validate_orderbook(
        [
            {"c_rate": 0.5, "blockchain_address": "0xA"},
            {"c_rate": 1.0, "blockchain_address": "0xB"},
        ],
        agent_addr="agent",
    )


@pytest.mark.parametrize("address", [None, "", "   ", 123])
def test_validate_orderbook_rejects_missing_blockchain_address(address):
    role = _make_role()
    order = {"bid_id": "s1", "c_rate": 1.0}
    if address is not None:
        order["blockchain_address"] = address

    with pytest.raises(ValueError, match="blockchain_address"):
        role.validate_orderbook([order], agent_addr="agent")


def test_blockchain_address_is_required_field():
    assert "blockchain_address" in BatteryClearing.required_fields


def test_validate_orderbook_rejects_node_outside_configured_locations():
    role = _make_role(locations=("A", "B"))
    with pytest.raises(ValueError, match="allowed locations"):
        role.validate_orderbook(
            [{"c_rate": 1.0, "location": "C", "blockchain_address": "0xA"}],
            agent_addr="agent",
        )


def test_clear_assigns_uniform_price_and_rejects_unawarded_orders_with_real_solver():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "blockchain_address": "0xs1", "volume": 5, "price": 12},
        {"bid_id": "s2", "blockchain_address": "0xs2", "volume": 4, "price": 15},
        {"bid_id": "d1", "blockchain_address": "0xd1", "volume": -3, "price": 20},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, flows = role.clear(orderbook, [product])

    assert {order["bid_id"] for order in accepted_orders} == {"s1", "d1"}
    assert {order["bid_id"] for order in rejected_orders} == {"s2"}
    assert all(order["accepted_price"] == 12 for order in accepted_orders)
    assert all(order["accepted_price"] == 0 for order in rejected_orders)
    assert meta[0]["supply_volume"] == 3
    assert meta[0]["demand_volume"] == 3
    assert flows == []


def test_clear_no_trade_with_real_solver_sets_zero_price():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "blockchain_address": "0xs1", "volume": 5, "price": 12},
        {"bid_id": "d1", "blockchain_address": "0xd1", "volume": -5, "price": 10},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, flows = role.clear(orderbook, [product])

    assert accepted_orders == []
    assert {order["bid_id"] for order in rejected_orders} == {"s1", "d1"}
    assert all(order["accepted_volume"] == 0 for order in rejected_orders)
    assert all(order["accepted_price"] == 0 for order in rejected_orders)
    assert meta[0]["supply_volume"] == 0
    assert meta[0]["demand_volume"] == 0
    assert flows == []


def test_clear_multi_location_sets_prices_and_meta_per_location():
    role = _make_role(locations=("A", "B"))
    orderbook = [
        # market A
        {
            "bid_id": "A_s1",
            "blockchain_address": "0xA_s1",
            "volume": 5,
            "price": 10,
            "location": "A",
        },
        {
            "bid_id": "A_s2",
            "blockchain_address": "0xA_s2",
            "volume": 5,
            "price": 13,
            "location": "A",
        },
        {
            "bid_id": "A_d1",
            "blockchain_address": "0xA_d1",
            "volume": -4,
            "price": 20,
            "location": "A",
        },
        # market B
        {
            "bid_id": "B_s1",
            "blockchain_address": "0xB_s1",
            "volume": 4,
            "price": 8,
            "location": "B",
        },
        {
            "bid_id": "B_s2",
            "blockchain_address": "0xB_s2",
            "volume": 3,
            "price": 12,
            "location": "B",
        },
        {
            "bid_id": "B_d1",
            "blockchain_address": "0xB_d1",
            "volume": -3,
            "price": 18,
            "location": "B",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])

    accepted_by_id = {o["bid_id"]: o for o in accepted_orders}
    rejected_ids = {o["bid_id"] for o in rejected_orders}
    meta_by_location = {m["location"]: m for m in meta}

    assert {"A_s1", "A_d1", "B_s1", "B_d1"} <= set(accepted_by_id)
    assert {"A_s2", "B_s2"} <= rejected_ids
    assert accepted_by_id["A_s1"]["accepted_price"] == 10
    assert accepted_by_id["A_d1"]["accepted_price"] == 10
    assert accepted_by_id["B_s1"]["accepted_price"] == 8
    assert accepted_by_id["B_d1"]["accepted_price"] == 8
    assert accepted_by_id["A_s1"]["accepted_volume"] > 0
    assert accepted_by_id["B_s1"]["accepted_volume"] > 0
    assert accepted_by_id["A_d1"]["accepted_volume"] < 0
    assert accepted_by_id["B_d1"]["accepted_volume"] < 0
    assert meta_by_location["A"]["supply_volume"] == 4
    assert meta_by_location["A"]["demand_volume"] == 4
    assert meta_by_location["B"]["supply_volume"] == 3
    assert meta_by_location["B"]["demand_volume"] == 3


def test_exclusive_bids_priority_A_d1_and_B_x():
    role = _make_role(locations=("A", "B"))
    orderbook = [
        # market A
        {
            "bid_id": "A_d1",
            "blockchain_address": "0xA_d1",
            "volume": -10,
            "price": 30,
            "location": "A",
        },
        {
            "bid_id": "A_local",
            "blockchain_address": "0xA_local",
            "volume": 10,
            "price": 10,
            "location": "A",
        },
        {
            "bid_id": "A_d2",
            "blockchain_address": "0xA_d2",
            "volume": -5,
            "price": 20,
            "location": "A",
        },  # this would be accepted if A_x would not be exclusive
        {
            "bid_id": "A_x",
            "blockchain_address": "0xA_x",
            "volume": 5,
            "price": 12,
            "location": "A",
            "exclusive_id": "X",
        },
        # market B
        {
            "bid_id": "B_d1",
            "blockchain_address": "0xB_d1",
            "volume": -5,
            "price": 40,
            "location": "B",
        },
        {
            "bid_id": "B_x",
            "blockchain_address": "0xB_x",
            "volume": 5,
            "price": 12,
            "location": "B",
            "exclusive_id": "X",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}
    meta_by_location = {m["location"]: m for m in meta}

    assert by_id["A_d1"]["accepted_volume"] == -10
    assert by_id["A_local"]["accepted_volume"] == 10
    assert by_id["A_x"]["accepted_volume"] == 0
    assert by_id["B_x"]["accepted_volume"] == 5
    assert by_id["A_d2"]["accepted_volume"] == 0
    assert by_id["B_d1"]["accepted_volume"] < 0
    assert (by_id["A_x"]["accepted_volume"] / 5) + (
        by_id["B_x"]["accepted_volume"] / 5
    ) <= 1.0 + 1e-9
    assert meta_by_location["A"]["price"] == 10
    assert meta_by_location["B"]["price"] == 12


def test_exclusive_bid_link_accepted_A():
    role = _make_role(locations=("A", "B"))
    orderbook = [
        {
            "bid_id": "D1",
            "blockchain_address": "0xD1",
            "volume": -10,
            "price": 10,
            "location": "A",
            "exclusive_id": "X",
        },
        {
            "bid_id": "S1",
            "blockchain_address": "0xS1",
            "volume": 10,
            "price": 5,
            "location": "A",
        },
        {
            "bid_id": "D2",
            "blockchain_address": "0xD2",
            "volume": -10,
            "price": 8,
            "location": "B",
            "exclusive_id": "X",
        },
        {
            "bid_id": "S2",
            "blockchain_address": "0xS2",
            "volume": 10,
            "price": 4,
            "location": "B",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == -10
    assert by_id["S1"]["accepted_volume"] == 10
    assert by_id["D2"]["accepted_volume"] == 0
    assert by_id["S2"]["accepted_volume"] == 0


def test_exclusive_bid_link_accepted_B():
    role = _make_role(locations=("A", "B"))
    orderbook = [
        {
            "bid_id": "D1",
            "blockchain_address": "0xD1",
            "volume": -10,
            "price": 10,
            "location": "A",
            "exclusive_id": "X",
        },
        {
            "bid_id": "S1",
            "blockchain_address": "0xS1",
            "volume": 10,
            "price": 7,
            "location": "A",
        },
        {
            "bid_id": "D2",
            "blockchain_address": "0xD2",
            "volume": -10,
            "price": 8,
            "location": "B",
            "exclusive_id": "X",
        },
        {
            "bid_id": "S2",
            "blockchain_address": "0xS2",
            "volume": 10,
            "price": 4,
            "location": "B",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == 0
    assert by_id["S1"]["accepted_volume"] == 0
    assert by_id["D2"]["accepted_volume"] == -10
    assert by_id["S2"]["accepted_volume"] == 10


def test_exclusive_bid_same_market_A():
    role = _make_role(locations=("A"))
    orderbook = [
        {
            "bid_id": "D1",
            "blockchain_address": "0xD1",
            "volume": -10,
            "price": 10,
            "location": "A",
            "exclusive_id": "X",
        },
        {
            "bid_id": "D2",
            "blockchain_address": "0xD2",
            "volume": -10,
            "price": 12,
            "location": "A",
            "exclusive_id": "X",
        },
        {
            "bid_id": "S1",
            "blockchain_address": "0xS1",
            "volume": 10,
            "price": 5,
            "location": "A",
        },
        {
            "bid_id": "S2",
            "blockchain_address": "0xS2",
            "volume": 10,
            "price": 4,
            "location": "A",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == 0
    assert by_id["D2"]["accepted_volume"] == -10
    assert by_id["S1"]["accepted_volume"] == 0
    assert by_id["S2"]["accepted_volume"] == 10


def test_clear_all_with_exclusive_missing():
    role = _make_role()
    orderbook = [
        {
            "bid_id": "D1",
            "blockchain_address": "0xD1",
            "volume": -10,
            "price": 10,
            "location": "A",
        },
        {
            "bid_id": "D2",
            "blockchain_address": "0xD2",
            "volume": -10,
            "price": 12,
            "location": "A",
        },
        {
            "bid_id": "S1",
            "blockchain_address": "0xS1",
            "volume": 10,
            "price": 5,
            "location": "A",
        },
        {
            "bid_id": "S2",
            "blockchain_address": "0xS2",
            "volume": 10,
            "price": 4,
            "location": "A",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == -10
    assert by_id["D2"]["accepted_volume"] == -10
    assert by_id["S1"]["accepted_volume"] == 10
    assert by_id["S2"]["accepted_volume"] == 10


def _matches(order):
    return [
        (m["bid_id"], m["blockchain_address"], m["volume"])
        for m in json.loads(order["matches"])
    ]


def test_clear_matches_cheapest_supply_with_highest_demand():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "volume": 5, "price": 10, "blockchain_address": "0xS1"},
        {"bid_id": "s2", "volume": 5, "price": 12, "blockchain_address": "0xS2"},
        {"bid_id": "s3", "volume": 5, "price": 30, "blockchain_address": "0xS3"},
        {"bid_id": "d1", "volume": -4, "price": 20, "blockchain_address": "0xD1"},
        {"bid_id": "d2", "volume": -3, "price": 25, "blockchain_address": "0xD2"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, _, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    # d2 pays most and gets the cheapest supply first
    assert _matches(by_id["s1"]) == [("d2", "0xD2", 3), ("d1", "0xD1", 2)]
    assert _matches(by_id["s2"]) == [("d1", "0xD1", 2)]
    assert _matches(by_id["d2"]) == [("s1", "0xS1", 3)]
    assert _matches(by_id["d1"]) == [("s1", "0xS1", 2), ("s2", "0xS2", 2)]
    assert _matches(by_id["s3"]) == []
    assert all(
        m["price"] == 12 for o in accepted_orders for m in json.loads(o["matches"])
    )


def test_clear_matched_volumes_add_up_to_accepted_volumes():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "volume": 2.5, "price": 1, "blockchain_address": "0xS1"},
        {"bid_id": "s2", "volume": 1.5, "price": 2, "blockchain_address": "0xS2"},
        {"bid_id": "d1", "volume": -1, "price": 5, "blockchain_address": "0xD1"},
        {"bid_id": "d2", "volume": -2, "price": 4, "blockchain_address": "0xD2"},
        {"bid_id": "d3", "volume": -1, "price": 3, "blockchain_address": "0xD3"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, _, _, _ = role.clear(orderbook, [product])

    for order in accepted_orders:
        matched = sum(m["volume"] for m in json.loads(order["matches"]))
        assert matched == pytest.approx(abs(order["accepted_volume"]))


def test_clear_matches_only_within_location():
    role = _make_role(locations=("A", "B"))
    orderbook = [
        {
            "bid_id": "A_s",
            "volume": 5,
            "price": 1,
            "location": "A",
            "blockchain_address": "0xAS",
        },
        {
            "bid_id": "A_d",
            "volume": -5,
            "price": 9,
            "location": "A",
            "blockchain_address": "0xAD",
        },
        {
            "bid_id": "B_s",
            "volume": 5,
            "price": 2,
            "location": "B",
            "blockchain_address": "0xBS",
        },
        {
            "bid_id": "B_d",
            "volume": -5,
            "price": 9,
            "location": "B",
            "blockchain_address": "0xBD",
        },
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, _, _, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders}

    assert _matches(by_id["A_s"]) == [("A_d", "0xAD", 5)]
    assert _matches(by_id["B_d"]) == [("B_s", "0xBS", 5)]


def test_clear_no_trade_sets_empty_matches():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "volume": 5, "price": 12, "blockchain_address": "0xS1"},
        {"bid_id": "d1", "volume": -5, "price": 10, "blockchain_address": "0xD1"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    _, rejected_orders, _, _ = role.clear(orderbook, [product])

    assert all(json.loads(o["matches"]) == [] for o in rejected_orders)


def _brp_orderbook():
    start, end = datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 8, 10, 0)
    return [
        {
            "bid_id": "S_01_1",
            "unit_id": "S_01",
            "volume": 5,
            "price": 10,
            "blockchain_address": "0xS1",
            "start_time": start,
            "end_time": end,
        },
        {
            "bid_id": "B_01_1",
            "unit_id": "B_01",
            "volume": -3,
            "price": 20,
            "blockchain_address": "0xB1",
            "start_time": start,
            "end_time": end,
        },
        {
            "bid_id": "B_02_1",
            "unit_id": "B_02",
            "volume": -3,
            "price": 5,
            "blockchain_address": "0xB2",
            "start_time": start,
            "end_time": end,
        },
    ]


def test_clear_does_not_send_market_results_without_brp_url(monkeypatch):
    monkeypatch.setattr(module, "BRP_MARKET_RESULTS_URL", None)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not send")
    )
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 8, 10, 0), None)

    accepted_orders, _, _, _ = _make_role().clear(_brp_orderbook(), [product])

    assert len(accepted_orders) == 2


def test_clear_sends_accepted_orders_to_brp(monkeypatch):
    sent = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        sent.append(request)
        return FakeResponse()

    monkeypatch.setattr(module, "BRP_MARKET_RESULTS_URL", "http://brp/market_results")
    monkeypatch.setattr(module, "MARKET_RESULTS_PASSWORD", "pw")
    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    role = _make_role()
    role.marketconfig.market_id = "BatteryMarket"
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 8, 10, 0), None)

    role.clear(_brp_orderbook(), [product])

    assert len(sent) == 1
    assert sent[0].full_url == "http://brp/market_results"
    assert (
        sent[0].get_header("Authorization")
        == "Basic " + base64.b64encode(b"assume:pw").decode()
    )
    payload = json.loads(sent[0].data)
    assert payload["market_id"] == "BatteryMarket"
    assert payload["start_time"] == "2026-01-01T10:00:00"
    assert payload["end_time"] == "2026-01-08T10:00:00"
    by_id = {o["bid_id"]: o for o in payload["orders"]}
    assert set(by_id) == {"S_01_1", "B_01_1"}
    assert by_id["S_01_1"]["start_time"] == "2026-01-01T10:00:00"
    assert by_id["S_01_1"]["matches"] == [
        {"bid_id": "B_01_1", "blockchain_address": "0xB1", "volume": 3, "price": 10}
    ]
    assert by_id["B_01_1"]["blockchain_address"] == "0xB1"


def test_clear_sends_delivery_period_to_brp_without_trades(monkeypatch):
    sent = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(module, "BRP_MARKET_RESULTS_URL", "http://brp/market_results")
    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    role = _make_role()
    role.marketconfig.market_id = "BatteryMarket"
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 8, 10, 0), None)
    orderbook = [
        {"bid_id": "s1", "volume": 5, "price": 12, "blockchain_address": "0xS1"},
        {"bid_id": "d1", "volume": -5, "price": 10, "blockchain_address": "0xD1"},
    ]

    accepted_orders, _, _, _ = role.clear(orderbook, [product])

    assert accepted_orders == []
    assert sent == [
        {
            "market_id": "BatteryMarket",
            "start_time": "2026-01-01T10:00:00",
            "end_time": "2026-01-08T10:00:00",
            "orders": [],
        }
    ]
