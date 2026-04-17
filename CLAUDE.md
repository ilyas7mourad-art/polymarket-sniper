# Instructions for Claude Code

## Project context
This is a Polymarket sniper bot that trades BTC/ETH 5-minute up/down markets.
The strategy is based on backtested edge: entry at 60-71¢ in the first 90 seconds
of a candle, after a 0.10%+ price move in one direction. Expected win rate ~72%.

Reference: `/Users/mourad/tennis_project/copybot/` contains the existing copy bot
and `polymarket_reference.txt` has full API docs.

## Workflow rules (CRITICAL)
1. NEVER commit directly to `main`. Always create a feature branch.
2. Branch naming: `feat/<description>`, `fix/<description>`, `refactor/<description>`.
3. Every change goes through a pull request. No exceptions.
4. When opening a PR, fill out the template in `.github/pull_request_template.md`.
5. After pushing, output the PR URL so the user can share it with the reviewer (Claude chat).
6. Do NOT merge PRs yourself. The user merges after review.
7. After a PR is merged, checkout main, pull, and delete the local feature branch.

## Code standards
- Python 3.11+
- Type hints on all public functions
- Docstrings on all modules, classes, public functions (Google style)
- Run `pytest` before pushing — don't push failing code
- No live orders, no real money in this phase (dev only)

## Current phase: DEV (no live trading)
All executor code must be a no-op or raise NotImplementedError.
Paper trading and backtesting only until explicitly told otherwise.
