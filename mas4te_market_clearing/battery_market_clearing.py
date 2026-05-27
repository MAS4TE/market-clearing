# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import timedelta
from operator import itemgetter

import pyomo.environ as pyo
from assume.common.market_objects import MarketConfig, Order, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


def calculate_meta(accepted_supply_orders, accepted_demand_orders, product, node=None):
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
        "node": node,
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
      the locations from the orders' ``node`` field (legacy single-market
      behaviour, where every order may have ``node=None``).
    - ``exclusive_link_field`` (optional, default ``"exclusive_id"``): name of
      the order field that identifies the exclusive group. Orders sharing the
      same value in this field are mutually exclusive.
    """

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
            if locations:
                node = order.get("node")
                if node not in locations:
                    raise ValueError(
                        f"Order's node {node!r} is not in allowed locations {locations}"
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
        provided. Otherwise infers them from the orders' ``node`` field, which
        keeps the legacy single-market behaviour (every order has
        ``node=None``) intact.
        """
        configured = self._configured_locations()
        if configured:
            return configured
        observed = {order.get("node") for order in orderbook}
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
        locations = self._resolve_locations(orderbook)

        # group orders by location and direction
        supply_orders_by_loc = defaultdict(list)
        demand_orders_by_loc = defaultdict(list)
        for order in orderbook:
            loc = order.get("node")
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
            loc_accepted = [o for o in accepted_orders if o.get("node") == loc]
            loc_rejected = [o for o in rejected_orders if o.get("node") == loc]
            loc_accepted_supply = [o for o in loc_accepted if o["accepted_volume"] > 0]
            loc_accepted_demand = [o for o in loc_accepted if o["accepted_volume"] < 0]

            if loc_accepted_supply:
                clear_price = float(max(map(itemgetter("price"), loc_accepted_supply)))
            else:
                clear_price = 0

            for order in loc_accepted:
                order["accepted_price"] = clear_price

            for order in loc_rejected:
                order["accepted_volume"] = 0
                order["accepted_price"] = clear_price

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
                    node=loc,
                )
            )

        flows = []

        return accepted_orders, rejected_orders, meta, flows
