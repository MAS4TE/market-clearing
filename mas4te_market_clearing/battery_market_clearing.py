# SPDX-FileCopyrightText: MAS4TE Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from operator import itemgetter

import pyomo.environ as pyo
from assume.common.market_objects import MarketConfig, Order, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


def calculate_meta(accepted_supply_orders, accepted_demand_orders, product):
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
        "node": None,
        "product_start": product[0],
        "product_end": product[1],
        "only_hours": product[2],
    }


class BatteryClearing(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def validate_orderbook(self, orderbook: Orderbook, agent_addr) -> None:
        allowed_c_rates = self.marketconfig.param_dict["allowed_c_rates"]
        for order in orderbook:
            if order["c_rate"] not in allowed_c_rates:
                raise ValueError(f"{order['c_rate']} is not in {allowed_c_rates}")

        super().validate_orderbook(orderbook, agent_addr)

    def set_model_restrictions(self, model: pyo.ConcreteModel) -> None:
        """Sets the model restrictions."""

        # Restrict the supply volume to be less than or equal to the demand volume
        model.restrict_product_balance = pyo.Constraint(
            expr=sum(model.supply_volume[i] for i in model.supply_volume)
            == sum(model.demand_volume[i] for i in model.demand_volume)
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

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        # get demand and supply orders from orderbook
        demand_orders = [x for x in orderbook if x["volume"] < 0]
        supply_orders = [x for x in orderbook if x["volume"] > 0]

        # create pyomo model for solving
        model = pyo.ConcreteModel()

        # add supply and demand variables to the model
        self.add_supply_vars(model, supply_orders)
        self.add_demand_vars(model, demand_orders)

        # add restrictions to the model
        self.set_model_restrictions(model)

        # set the model objective
        self.set_model_objective(model)

        # run optimization (clearing)
        self.solve_model(model, solver="highs")

        # get accepted orders
        accepted_orders, rejected_orders = self.get_accepted_rejected_orders(
            model, supply_orders, demand_orders
        )

        accepted_demand_orders = [
            x for x in accepted_orders if x["accepted_volume"] < 0
        ]
        accepted_supply_orders = [
            x for x in accepted_orders if x["accepted_volume"] > 0
        ]

        # use uniform pricing
        if accepted_orders:
            clear_price = float(max(map(itemgetter("price"), accepted_supply_orders)))
        else:
            clear_price = 0

        for order in accepted_orders:
            order["accepted_price"] = clear_price

        # set accepted volume to 0 and price to clear price for rejected orders
        for order in rejected_orders:
            order["accepted_volume"] = 0
            order["accepted_price"] = clear_price

        print(f"Clearing price: {clear_price * 100} ct./kWh")
        print(
            f"{sum([abs(x['accepted_volume']) for x in accepted_demand_orders])} kWh traded volume"
        )
        print("----------------------------------------")

        meta = []

        meta.append(
            calculate_meta(
                accepted_supply_orders,
                accepted_demand_orders,
                market_products[0],
            )
        )
        flows = []

        return accepted_orders, rejected_orders, meta, flows
