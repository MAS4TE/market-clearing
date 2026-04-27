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
  (e.g. supply/demand volume, uniform price, product interval, `node`).
- `flows`: Currently an empty list.

## Multi-Market / Exclusive Bids

`BatteryClearing` can jointly clear several markets (e.g. one at location `A`
and one at location `B`) in a single optimization run. Each location keeps its
own supply/demand balance and its own uniform clearing price; orders are
assigned to a market via the standard `node` field of `Order`.

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
| `locations` | optional | List of locations handled by this clearing instance, e.g. `["A", "B"]`. If omitted, the clearing infers locations from the orders' `node` field (legacy single-market mode). |
| `exclusive_link_field` | optional | Name of the order field identifying the exclusive group. Defaults to `"exclusive_id"`. |

### Required `Order` fields

In addition to the standard ASSUME `Order` fields:

- `node`: location identifier (e.g. `"A"` or `"B"`).
- `c_rate`: as before.
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
