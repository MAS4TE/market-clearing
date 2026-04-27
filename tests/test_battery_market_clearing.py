import importlib
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
    for solver_name in ("gurobi", "highs", "appsi_highs", "glpk", "cbc"):
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
    role.validate_orderbook([{"c_rate": 0.5}, {"c_rate": 1.0}], agent_addr="agent")


def test_validate_orderbook_rejects_node_outside_configured_locations():
    role = _make_role(locations=("A", "B"))
    with pytest.raises(ValueError, match="allowed locations"):
        role.validate_orderbook(
            [{"c_rate": 1.0, "node": "C"}],
            agent_addr="agent",
        )


def test_clear_assigns_uniform_price_and_rejects_unawarded_orders_with_real_solver():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "volume": 5, "price": 12},
        {"bid_id": "s2", "volume": 4, "price": 15},
        {"bid_id": "d1", "volume": -3, "price": 20},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, flows = role.clear(orderbook, [product])

    assert {order["bid_id"] for order in accepted_orders} == {"s1", "d1"}
    assert {order["bid_id"] for order in rejected_orders} == {"s2"}
    assert all(
        order["accepted_price"] == 12 for order in accepted_orders + rejected_orders
    )
    assert meta[0]["supply_volume"] == 3
    assert meta[0]["demand_volume"] == 3
    assert flows == []


def test_clear_no_trade_with_real_solver_sets_zero_price():
    role = _make_role()
    orderbook = [
        {"bid_id": "s1", "volume": 5, "price": 12},
        {"bid_id": "d1", "volume": -5, "price": 10},
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
        {"bid_id": "A_s1", "volume": 5, "price": 10, "node": "A"},
        {"bid_id": "A_s2", "volume": 5, "price": 13, "node": "A"},
        {"bid_id": "A_d1", "volume": -4, "price": 20, "node": "A"},
        # market B
        {"bid_id": "B_s1", "volume": 4, "price": 8, "node": "B"},
        {"bid_id": "B_s2", "volume": 3, "price": 12, "node": "B"},
        {"bid_id": "B_d1", "volume": -3, "price": 18, "node": "B"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])

    accepted_by_id = {o["bid_id"]: o for o in accepted_orders}
    rejected_ids = {o["bid_id"] for o in rejected_orders}
    meta_by_node = {m["node"]: m for m in meta}

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
    assert meta_by_node["A"]["supply_volume"] == 4
    assert meta_by_node["A"]["demand_volume"] == 4
    assert meta_by_node["B"]["supply_volume"] == 3
    assert meta_by_node["B"]["demand_volume"] == 3


def test_clear_cross_border_exclusive_bid_links_markets():
    role = _make_role(locations=("A", "B"))
    orderbook = [
        # market A
        {"bid_id": "A_d1", "volume": -10, "price": 30, "node": "A"},
        {"bid_id": "A_local", "volume": 10, "price": 10, "node": "A"},
        {
            "bid_id": "A_d2",
            "volume": -5,
            "price": 20,
            "node": "A",
        },  # this would be accepted if A_x would not be exclusive
        {"bid_id": "A_x", "volume": 5, "price": 12, "node": "A", "exclusive_id": "X"},
        # market B
        {"bid_id": "B_d1", "volume": -5, "price": 40, "node": "B"},
        {"bid_id": "B_x", "volume": 5, "price": 12, "node": "B", "exclusive_id": "X"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)

    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}
    meta_by_node = {m["node"]: m for m in meta}

    assert by_id["A_d1"]["accepted_volume"] == -10
    assert by_id["A_local"]["accepted_volume"] == 10

    assert by_id["A_x"]["accepted_volume"] == 0
    assert by_id["B_x"]["accepted_volume"] == 5

    assert by_id["A_d2"]["accepted_volume"] == 0
    assert by_id["B_d1"]["accepted_volume"] < 0
    assert (by_id["A_x"]["accepted_volume"] / 5) + (
        by_id["B_x"]["accepted_volume"] / 5
    ) <= 1.0 + 1e-9
    assert meta_by_node["A"]["price"] == 10
    assert meta_by_node["B"]["price"] == 12

    role = _make_role(locations=("A", "B"))
    orderbook = [
        {"bid_id": "D1", "volume": -10, "price": 10, "node": "A", "exclusive_id": "X"},
        {"bid_id": "S1", "volume": 10, "price": 5, "node": "A"},
        {"bid_id": "D2", "volume": -10, "price": 8, "node": "B", "exclusive_id": "X"},
        {"bid_id": "S2", "volume": 10, "price": 4, "node": "B"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == -10
    assert by_id["S1"]["accepted_volume"] == 10
    assert by_id["D2"]["accepted_volume"] == 0
    assert by_id["S2"]["accepted_volume"] == 0

    role = _make_role(locations=("A", "B"))
    orderbook = [
        {"bid_id": "D1", "volume": -10, "price": 10, "node": "A", "exclusive_id": "X"},
        {"bid_id": "S1", "volume": 10, "price": 7, "node": "A"},
        {"bid_id": "D2", "volume": -10, "price": 8, "node": "B", "exclusive_id": "X"},
        {"bid_id": "S2", "volume": 10, "price": 4, "node": "B"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == 0
    assert by_id["S1"]["accepted_volume"] == 0
    assert by_id["D2"]["accepted_volume"] == -10
    assert by_id["S2"]["accepted_volume"] == 10

    role = _make_role(locations=("A"))
    orderbook = [
        {"bid_id": "D1", "volume": -10, "price": 10, "node": "A", "exclusive_id": "X"},
        {"bid_id": "D2", "volume": -10, "price": 12, "node": "A", "exclusive_id": "X"},
        {"bid_id": "S1", "volume": 10, "price": 5, "node": "A"},
    ]
    product = (datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0), None)
    accepted_orders, rejected_orders, meta, _ = role.clear(orderbook, [product])
    by_id = {o["bid_id"]: o for o in accepted_orders + rejected_orders}

    assert by_id["D1"]["accepted_volume"] == 0
    assert by_id["D2"]["accepted_volume"] == -10
    assert by_id["S1"]["accepted_volume"] == 10
