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
- `meta`: List of aggregated metrics (e.g. supply/demand volume, price, product interval).
- `flows`: Currently an empty list.

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
