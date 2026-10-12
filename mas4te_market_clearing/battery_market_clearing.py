# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import base64
import json
import logging
import os
import urllib.request
from collections import defaultdict
from datetime import timedelta
from operator import itemgetter

import pyomo.environ as pyo
from assume.common.market_objects import MarketConfig, Order, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)

# order field carrying the bidder's blockchain address (required on every order)
BLOCKCHAIN_ADDRESS_FIELD = "blockchain_address"
# order field the clearing fills with the matched counterparties (JSON string)
MATCHES_FIELD = "matches"
# volumes below this are treated as zero when splitting matches
VOLUME_EPS = 1e-9
# BRP endpoint for the market results, nothing is sent if it is not set
BRP_MARKET_RESULTS_URL = os.getenv("BRP_MARKET_RESULTS_URL")
BRP_API_PASSWORD = os.getenv("BRP_API_PASSWORD")


def validate_unique_bid_ids(orderbook: Orderbook) -> None:
    """Reject orderbooks where ``bid_id`` is reused.

    Supply/demand variables and prices are indexed only by ``bid_id`` in the
    Pyomo model; duplicates silently merge orders and corrupt the optimum.
    """
    seen: set[object] = set()
    duplicate_ids: set[object] = set()
    for order in orderbook:
        bid_id = order["bid_id"]
        if bid_id in seen:
            duplicate_ids.add(bid_id)
        seen.add(bid_id)
    if duplicate_ids:
        # set order is arbitrary; stringify for a readable message
        names = ", ".join(str(bid_id) for bid_id in sorted(duplicate_ids, key=str))
        raise ValueError(
            f"Same bid_id used more than once: {names}. "
            "Give every order its own bid_id, otherwise several orders collapse into one variable."
        )


def match_counterparties(
    accepted_supply_orders: list[Order],
    accepted_demand_orders: list[Order],
    clear_price: float,
) -> None:
    """Assign each accepted order the counterparties it trades with.

    The clearing itself only balances total supply and demand, so the pairing
    is derived afterwards: the cheapest supply is matched with the
    highest-paying demand until the accepted volumes are used up (ties are
    broken by ``bid_id``). Every match carries the counterparty's
    ``bid_id``, blockchain address, the matched volume and the uniform price.

    The matches are stored as JSON string in ``order["matches"]`` because
    ASSUME splits orders with dict values and cannot write lists to the
    database.
    """
    supply_queue = sorted(
        accepted_supply_orders, key=lambda o: (o["price"], str(o["bid_id"]))
    )
    demand_queue = sorted(
        accepted_demand_orders, key=lambda o: (-o["price"], str(o["bid_id"]))
    )
    remaining = {
        id(order): abs(order["accepted_volume"])
        for order in supply_queue + demand_queue
    }
    matches = {id(order): [] for order in supply_queue + demand_queue}

    s_idx = d_idx = 0
    while s_idx < len(supply_queue) and d_idx < len(demand_queue):
        supply = supply_queue[s_idx]
        demand = demand_queue[d_idx]
        volume = min(remaining[id(supply)], remaining[id(demand)])

        if volume > VOLUME_EPS:
            for order, counterpart in ((supply, demand), (demand, supply)):
                matches[id(order)].append(
                    {
                        "bid_id": counterpart["bid_id"],
                        BLOCKCHAIN_ADDRESS_FIELD: counterpart[BLOCKCHAIN_ADDRESS_FIELD],
                        "volume": volume,
                        "price": clear_price,
                    }
                )
            remaining[id(supply)] -= volume
            remaining[id(demand)] -= volume

        if remaining[id(supply)] <= VOLUME_EPS:
            s_idx += 1
        if remaining[id(demand)] <= VOLUME_EPS:
            d_idx += 1

    for order in supply_queue + demand_queue:
        order[MATCHES_FIELD] = json.dumps(matches[id(order)], default=str)


def send_market_results(
    market_id: str, product: tuple, accepted_orders: list[Order]
) -> None:
    """Report the accepted orders to the BRP, so it knows who was awarded how
    much and with whom.

    The delivery period is always sent, also without accepted orders, so the BRP
    can replace an earlier result for it (e.g. from a re-run of the simulation).
    """
    payload = {
        "market_id": market_id,
        "start_time": product[0].isoformat(),
        "end_time": product[1].isoformat(),
        "orders": [
            {
                "bid_id": order["bid_id"],
                "unit_id": order["unit_id"],
                BLOCKCHAIN_ADDRESS_FIELD: order[BLOCKCHAIN_ADDRESS_FIELD],
                "start_time": order["start_time"].isoformat(),
                "end_time": order["end_time"].isoformat(),
                "accepted_volume": order["accepted_volume"],
                "accepted_price": order["accepted_price"],
                MATCHES_FIELD: json.loads(order[MATCHES_FIELD]),
            }
            for order in accepted_orders
        ],
    }

    credentials = base64.b64encode(f"assume:{BRP_API_PASSWORD}".encode()).decode()
    request = urllib.request.Request(
        BRP_MARKET_RESULTS_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {credentials}",
        },
        method="POST",
    )

    # a BRP that is down must not stop the simulation
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            logger.info(
                "Sent %d accepted orders of market '%s' to the BRP: %s",
                len(accepted_orders),
                market_id,
                response.status,
            )
    except Exception as e:
        logger.error(
            "Could not send market results of market '%s' to the BRP: %s",
            market_id,
            e,
        )


def calculate_meta(
    accepted_supply_orders,
    accepted_demand_orders,
    product,
    location=None,
):
    supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
    demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
    prices = list(map(itemgetter("accepted_price"), accepted_supply_orders)) or [0]
    # can also be self.marketconfig.maximum_bid..?
    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    avg_price = 0
    if supply_volume:
        weighted_price = [
            order["accepted_volume"] * order["accepted_price"]
            for order in accepted_supply_orders
        ]
        avg_price = sum(weighted_price) / supply_volume
    return {
        "supply_volume": supply_volume,
        "demand_volume": demand_volume,
        "demand_volume_energy": demand_volume * duration_hours,
        "supply_volume_energy": supply_volume * duration_hours,
        "price": avg_price,
        "max_price": max(prices),
        "min_price": min(prices),
        "location": location,
        "product_start": product[0],
        "product_end": product[1],
        "only_hours": product[2],
    }


class BatteryClearing(MarketRole):
    """Joint market clearing for one or more battery markets (locations).

    The clearing can simultaneously clear several locations (e.g. a market at
    location ``A`` and one at location ``B``) and supports cross-market
    *exclusive* bids: orders that share the same exclusive-group identifier
    will jointly satisfy ``sum(accepted_volume / volume) <= 1`` so that, in
    particular, a bid that is fully accepted on market A will be fully
    rejected on market B.

    Configuration via ``MarketConfig.param_dict``:

    - ``allowed_c_rates``: existing whitelist of permitted C-rates.
    - ``locations`` (optional): list of location identifiers handled by this
      clearing instance, e.g. ``["A", "B"]``. If omitted, the clearing infers
      the locations from the orders' ``location`` field (legacy single-market
      behaviour, where every order may have ``location=None``).
    - ``exclusive_link_field`` (optional, default ``"exclusive_id"``): name of
      the order field that identifies the exclusive group. Orders sharing the
      same value in this field are mutually exclusive.
    """

    # ASSUME checks that these are listed in ``MarketConfig.additional_fields``
    # and present on every order
    required_fields = ["c_rate", BLOCKCHAIN_ADDRESS_FIELD]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def _configured_locations(self) -> list:
        return list(self.marketconfig.param_dict.get("locations", []) or [])

    def _exclusive_link_field(self) -> str:
        return self.marketconfig.param_dict.get("exclusive_link_field", "exclusive_id")

    def validate_orderbook(self, orderbook: Orderbook, agent_addr) -> None:
        allowed_c_rates = self.marketconfig.param_dict["allowed_c_rates"]
        locations = self._configured_locations()
        for order in orderbook:
            if order["c_rate"] not in allowed_c_rates:
                raise ValueError(f"{order['c_rate']} is not in {allowed_c_rates}")
            address = order.get(BLOCKCHAIN_ADDRESS_FIELD)
            if not isinstance(address, str) or not address.strip():
                raise ValueError(
                    f"Order {order.get('bid_id')!r} has no valid {BLOCKCHAIN_ADDRESS_FIELD}: {address!r}"
                )
            if locations:
                location = _order_location(order)
                if location not in locations:
                    raise ValueError(
                        f"Order's location {location!r} is not in allowed locations {locations}"
                    )

        super().validate_orderbook(orderbook, agent_addr)

    def set_model_restrictions(
        self,
        model: pyo.ConcreteModel,
        supply_orders_by_loc: dict[str, list[Order]],
        demand_orders_by_loc: dict[str, list[Order]],
        exclusive_groups: dict[str, list[Order]],
    ) -> None:
        """Sets the model restrictions.

        Adds one supply==demand balance constraint per location and one
        exclusivity constraint per non-trivial exclusive group.

        Args:
            model: The Pyomo model to add the restrictions to.
            supply_orders_by_loc: A dictionary of supply orders by location with location as key.
            demand_orders_by_loc: A dictionary of demand orders by location with location as key.
            exclusive_groups: A dictionary of exclusive groups by group id with group id as key.
        """

        locations = sorted(
            set(supply_orders_by_loc) | set(demand_orders_by_loc),
            key=lambda x: (x is None, x),
        )
        model.locations = pyo.Set(initialize=locations, ordered=True)

        def _balance_rule(m, loc):
            supply_ids = [o["bid_id"] for o in supply_orders_by_loc.get(loc, [])]
            demand_ids = [o["bid_id"] for o in demand_orders_by_loc.get(loc, [])]
            if not supply_ids and not demand_ids:
                return pyo.Constraint.Skip
            return sum(m.supply_volume[i] for i in supply_ids) == sum(
                m.demand_volume[i] for i in demand_ids
            )

        model.restrict_product_balance = pyo.Constraint(
            model.locations, rule=_balance_rule
        )

        if exclusive_groups:
            group_ids = list(exclusive_groups.keys())
            model.exclusive_groups = pyo.Set(initialize=group_ids, ordered=True)

            def _exclusivity_rule(m, gid):
                terms = []
                for order in exclusive_groups[gid]:
                    max_volume = abs(order["volume"])
                    if max_volume == 0:
                        continue
                    var = (
                        m.supply_volume[order["bid_id"]]
                        if order["volume"] > 0
                        else m.demand_volume[order["bid_id"]]
                    )
                    terms.append(var / max_volume)
                if not terms:
                    return pyo.Constraint.Skip
                return sum(terms) <= 1

            model.restrict_exclusive_groups = pyo.Constraint(
                model.exclusive_groups, rule=_exclusivity_rule
            )

    def set_model_objective(self, model: pyo.ConcreteModel) -> None:
        """Sets the model objective function."""

        # calculate supply and demand costs
        supply_costs = sum(
            model.supply_price[bid_id] * model.supply_volume[bid_id]
            for bid_id in model.supply_price
        )
        demand_costs = sum(
            model.demand_price[bid_id] * model.demand_volume[bid_id]
            for bid_id in model.demand_price
        )

        # calculate the traded volume
        traded_volume = sum(model.supply_volume[i] for i in model.supply_volume)

        # maximize value (demand - supply costs) and add small
        # positive amount for each traded unit as incentive for equal prices
        model.objective = pyo.Objective(
            expr=demand_costs - supply_costs + 1e-6 * traded_volume, sense=pyo.maximize
        )

    def add_supply_vars(
        self, model: pyo.ConcreteModel, supply_orders: list[Order]
    ) -> None:
        """Creates supply price & volume variable."""

        model.supply_volume = pyo.Var(
            [supply_order["bid_id"] for supply_order in supply_orders],
            domain=pyo.NonNegativeReals,
        )
        for supply_order in supply_orders:
            max_volume = abs(supply_order["volume"])
            model.supply_volume[supply_order["bid_id"]].setub(max_volume)

        model.supply_price = pyo.Param(
            [supply_order["bid_id"] for supply_order in supply_orders],
            initialize={
                supply_order["bid_id"]: supply_order["price"]
                for supply_order in supply_orders
            },
            domain=pyo.NonNegativeReals,
        )

    def add_demand_vars(
        self, model: pyo.ConcreteModel, demand_orders: list[Order]
    ) -> None:
        """Creates demand price & volume variable."""

        model.demand_volume = pyo.Var(
            [demand_order["bid_id"] for demand_order in demand_orders],
            domain=pyo.NonNegativeReals,
        )
        for demand_order in demand_orders:
            max_volume = abs(demand_order["volume"])
            model.demand_volume[demand_order["bid_id"]].setub(max_volume)

        model.demand_price = pyo.Param(
            [demand_order["bid_id"] for demand_order in demand_orders],
            initialize={
                demand_order["bid_id"]: demand_order["price"]
                for demand_order in demand_orders
            },
            domain=pyo.NonNegativeReals,
        )

    def solve_model(self, model: pyo.ConcreteModel, solver: str = "highs") -> None:
        """Solves the model using the specified solver."""

        solver = pyo.SolverFactory(solver)
        results = solver.solve(model, tee=False)

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise ValueError("Model could not be solved optimally.")

        # Log the results
        logger.info("Model solved successfully.")
        logger.debug(f"Objective value: {pyo.value(model.objective)}")

    def get_accepted_rejected_orders(
        self,
        model: pyo.ConcreteModel,
        supply_orders: list[Order],
        demand_orders: list[Order],
    ) -> tuple[list[Order], list[Order]]:
        accepted_orders = []
        rejected_orders = []
        for supply_order in supply_orders:
            volume = model.supply_volume[supply_order["bid_id"]].value
            supply_order["accepted_volume"] = volume

            if volume > 0:
                accepted_orders.append(supply_order)
            else:
                rejected_orders.append(supply_order)

        for demand_order in demand_orders:
            volume = model.demand_volume[demand_order["bid_id"]].value
            demand_order["accepted_volume"] = -volume

            if volume > 0:
                accepted_orders.append(demand_order)
            else:
                rejected_orders.append(demand_order)

        return accepted_orders, rejected_orders

    def calculate_clearing_price(self) -> float:
        """Calculates the clearing price as the highest price of awarded asks or bids.

        Returns:
            float: The clearing price.
        """
        # Get all awarded ask prices where volume > 0
        awarded_ask_prices = [
            ask.price
            for ask in self.asks
            if self.model.ask_volume[ask.uuid, ask.product.id].value > 0
        ]

        # Get all awarded bid prices where volume > 0
        awarded_bid_prices = [
            bid.price
            for bid in self.bids
            if self.model.bid_volume[bid.uuid, bid.product.id].value > 0
        ]

        # Combine all awarded prices
        all_awarded_prices = awarded_ask_prices + awarded_bid_prices

        if not all_awarded_prices:
            raise ValueError("No trades occurred, clearing price cannot be determined.")

        # Return the highest awarded price
        return max(all_awarded_prices)

    def _resolve_locations(self, orderbook: Orderbook) -> list:
        """Determine which locations the current clearing run covers.

        Uses the configured ``locations`` from ``MarketConfig.param_dict`` if
        provided. Otherwise infers them from the orders' ``location`` field, which
        keeps the legacy single-market behaviour (every order has
        ``location=None``) intact.
        """
        configured = self._configured_locations()
        if configured:
            return configured
        observed = {_order_location(order) for order in orderbook}
        if observed:
            # Stable order: non-None first, then None at the end.
            return sorted(observed, key=lambda x: (x is None, x))
        return [None]

    def _collect_exclusive_groups(
        self, orderbook: Orderbook
    ) -> dict[object, list[Order]]:
        """Collect orders into exclusive groups.

        A group is only retained if it contains at least two orders, since a
        single-order group has no coupling effect.
        """
        field_name = self._exclusive_link_field()
        groups = defaultdict(list)
        for order in orderbook:
            group_id = order.get(field_name)
            if group_id is None:
                continue
            groups[group_id].append(order)
        return {gid: orders for gid, orders in groups.items() if len(orders) > 1}

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        validate_unique_bid_ids(orderbook)

        locations = self._resolve_locations(orderbook)

        # group orders by location and direction
        supply_orders_by_loc = defaultdict(list)
        demand_orders_by_loc = defaultdict(list)
        for order in orderbook:
            loc = _order_location(order)
            if loc not in locations:
                raise ValueError(
                    f"Order has unknown location {loc!r}; allowed: {locations}"
                )
            if order["volume"] > 0:
                supply_orders_by_loc[loc].append(order)
            elif order["volume"] < 0:
                demand_orders_by_loc[loc].append(order)

        all_supply_orders = [
            order for orders in supply_orders_by_loc.values() for order in orders
        ]
        all_demand_orders = [
            order for orders in demand_orders_by_loc.values() for order in orders
        ]

        # collect exclusive groups across markets (e.g. one bid on A linked to
        # its mirror bid on B)
        exclusive_groups = self._collect_exclusive_groups(orderbook)

        # create pyomo model for solving
        model = pyo.ConcreteModel()

        # add supply and demand variables to the model (flat over all
        # locations because bid_ids are globally unique)
        self.add_supply_vars(model, all_supply_orders)
        self.add_demand_vars(model, all_demand_orders)

        # add restrictions to the model: per-location balance + exclusivity
        self.set_model_restrictions(
            model,
            supply_orders_by_loc,
            demand_orders_by_loc,
            exclusive_groups,
        )

        # set the model objective (joint welfare across all markets)
        self.set_model_objective(model)

        # run optimization (joint clearing)
        self.solve_model(model, solver="highs")

        # get accepted orders
        accepted_orders, rejected_orders = self.get_accepted_rejected_orders(
            model, all_supply_orders, all_demand_orders
        )

        # uniform pricing per location (each market clears at its own price)
        meta = []
        for loc in locations:
            loc_accepted = [o for o in accepted_orders if _order_location(o) == loc]
            loc_rejected = [o for o in rejected_orders if _order_location(o) == loc]
            loc_accepted_supply = [o for o in loc_accepted if o["accepted_volume"] > 0]
            loc_accepted_demand = [o for o in loc_accepted if o["accepted_volume"] < 0]

            if loc_accepted_supply:
                clear_price = float(max(map(itemgetter("price"), loc_accepted_supply)))
            else:
                clear_price = 0

            for order in loc_accepted:
                order["accepted_price"] = clear_price

            match_counterparties(loc_accepted_supply, loc_accepted_demand, clear_price)

            for order in loc_rejected:
                order["accepted_volume"] = 0
                order["accepted_price"] = 0.0
                order[MATCHES_FIELD] = json.dumps([])

            label = f"[{loc}] " if loc is not None else ""
            print(f"{label}Clearing price: {clear_price * 100} ct./kWh")
            print(
                f"{label}{sum(abs(o['accepted_volume']) for o in loc_accepted_demand)} kWh traded volume"
            )
            print("----------------------------------------")

            meta.append(
                calculate_meta(
                    loc_accepted_supply,
                    loc_accepted_demand,
                    market_products[0],
                    location=loc,
                )
            )

        flows = []

        if BRP_MARKET_RESULTS_URL:
            send_market_results(
                self.marketconfig.market_id, market_products[0], accepted_orders
            )

        return accepted_orders, rejected_orders, meta, flows


def _order_location(order: Order) -> object:
    """Return the location identifier for an order.

    Accept both the new ``location`` field and the legacy ``node`` field so
    existing orderbooks keep working.
    """

    if "location" in order:
        return order.get("location")
    return order.get("node")
