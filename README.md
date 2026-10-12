# MAS4TE Market Clearing

Market clearing mechanism for battery-related order books in the MAS4TE context.

## Features

- Validates orders against allowed C-rates (`allowed_c_rates`).
- Performs market clearing via optimization (Pyomo + HiGHS).
- Applies uniform pricing for accepted orders.
- Returns accepted/rejected orders including meta information.

## Requirements

- Python `>=3.10`
- An installable solver for Pyomo (default in code: `highs`)


## Installation

### Option A: Install package from git

```bash
pip install git+https://github.com/MAS4TE/market-clearing
```

### Option B: Install package locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option C: Install dependencies from `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Clearing Return Values

- `accepted_orders`: Orders with `accepted_volume != 0`, including `accepted_price`.
- `rejected_orders`: Orders with `accepted_volume == 0`, including `accepted_price`.
- `meta`: List of aggregated metrics, **one entry per cleared location**
  (e.g. supply/demand volume, uniform price, product interval, `location`).
- `flows`: Currently an empty list.

## Blockchain Addresses and Matches

Every order must carry the bidder's blockchain address in the field
`blockchain_address` (non-empty string). `BatteryClearing.required_fields`
contains it, so ASSUME refuses a market config that does not list it in
`additional_fields`, and `validate_orderbook` rejects orders without one.

The optimization only balances total supply and demand per location, so who
trades with whom is derived afterwards: per location, the cheapest accepted
supply is matched with the highest-paying accepted demand until the accepted
volumes are used up (ties broken by `bid_id`). Each order then gets a field
`matches`, a **JSON string** (ASSUME splits orders with dict values and cannot
store lists in the database) holding a list of counterparties:

```json
[{"bid_id": "S_01_1", "blockchain_address": "0x...", "volume": 2.0, "price": 0.12}]
```

`volume` is the matched (positive) volume, `price` the uniform clearing price
of the location. Rejected orders get `"[]"`.

## Reporting to the BRP

If the environment variable `BRP_MARKET_RESULTS_URL` is set, `clear()` sends the delivery
period of the product (`start_time`, `end_time`) and the accepted orders (with `unit_id`,
`blockchain_address`, delivery period, accepted volume and price and the `matches`) as JSON
to that URL, using HTTP Basic Auth with the password from `BRP_API_PASSWORD`. The delivery
period is sent also if no order was accepted, so the BRP can replace an earlier result for
it. If it is not set, nothing is sent. A BRP that cannot be reached is
logged as an error and does not stop the clearing.

```bash
export BRP_MARKET_RESULTS_URL=https://mas4te.nowum.fh-aachen.de/brp-api/market_results
export BRP_API_PASSWORD=...
```

## Multi-Market / Exclusive Bids

`BatteryClearing` can jointly clear several markets (e.g. one at location `A`
and one at location `B`) in a single optimization run. Each location keeps its
own supply/demand balance and its own uniform clearing price; orders are
assigned to a market via the `location` field of `Order` (legacy: `node`).

Bids on different markets can be linked to be **mutually exclusive**: e.g. a
bidder may place the same offer on market `A` and on market `B` and require
that acceptance on one excludes acceptance on the other. Linked orders share
the same value in the order field configured via `exclusive_link_field`
(default: `exclusive_id`). The clearing then enforces

```text
sum(accepted_volume_o / volume_o for o in group) <= 1
```

so that, in particular, full acceptance on one market forces the linked
order(s) to be rejected. (Note: the constraint is LP-relaxed — fractional
splits between markets are feasible. Strict binary "either / or" exclusivity
would require a MIP solver.)

### Required `MarketConfig.param_dict` keys

| Key | Required | Description |
| --- | --- | --- |
| `allowed_c_rates` | yes | Whitelist of permitted C-rates per order. |
| `locations` | optional | List of locations handled by this clearing instance, e.g. `["A", "B"]`. If omitted, the clearing infers locations from the orders' `location` field (legacy: `node`, single-market mode). |
| `exclusive_link_field` | optional | Name of the order field identifying the exclusive group. Defaults to `"exclusive_id"`. |

### Required `Order` fields

In addition to the standard ASSUME `Order` fields:

- `location`: location identifier (e.g. `"A"` or `"B"`). (`node` is accepted as legacy alias.)
- `c_rate`: as before.
- `blockchain_address`: the bidder's blockchain address.
- `exclusive_id` (or whatever `exclusive_link_field` points at): identifier
  shared by all orders in the same exclusive group. Set to `None` (or omit) if
  the order is not coupled to any other.

## Project Structure

```text
mas4te_market_clearing/
  __init__.py
  battery_market_clearing.py
pyproject.toml
requirements.txt
```

## License

`MIT`
