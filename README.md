# polymarket-sniper

Sniper bot for BTC/ETH 5-minute up/down markets on Polymarket.

**Status: DEV phase — no live trading.**

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run tests

```bash
pytest
```

## Contributing

See [CLAUDE.md](CLAUDE.md) for contributor instructions and workflow rules.

## Roadmap

| PR | Description |
|----|-------------|
| #1 | Initial project scaffold (this PR) |
| #2 | Market scanner — fetch eligible 5-min BTC/ETH markets from Gamma API |
| #3 | Price feed — Binance WebSocket stream for BTC/ETH spot price |
| #4 | Signal engine — detect 0.10%+ momentum in first 90s of candle |
| #5 | Executor — paper trading order submission via CLOB API |
| #6 | Position manager — track open positions, exit logic, P&L logging |
