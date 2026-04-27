"""Track Bonereaper's trades in real-time via Polymarket data API.

Polls every POLL_INTERVAL seconds. Deduplicates by transactionHash.
Rotates CSV daily.
"""
import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

DATA_DIR = Path('/home/mma/polymarket-sniper/data')
WALLET = '0xeebde7a0e019a63e6b476eb425505b7b3e6eba30'
POLL_INTERVAL = 3
MAX_SEEN_CACHE = 50_000

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

HEADER = [
    'fetched_at_utc', 'trade_timestamp_utc', 'tx_hash', 'side',
    'asset_token_id', 'condition_id', 'outcome', 'outcome_index',
    'price', 'size', 'notional', 'market_slug', 'title'
]


def get_csv_path() -> Path:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return DATA_DIR / f'bonereaper_live_{today}.csv'


async def run() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen_tx: set[str] = set()
    current_path = None
    fh = None
    writer = None

    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            try:
                new_path = get_csv_path()
                if new_path != current_path:
                    if fh:
                        fh.close()
                    is_new = not new_path.exists()
                    fh = open(new_path, 'a', newline='')
                    writer = csv.writer(fh)
                    if is_new:
                        writer.writerow(HEADER)
                    current_path = new_path
                    logger.info(f'Writing to {new_path}')

                r = await client.get(
                    'https://data-api.polymarket.com/trades',
                    params={'user': WALLET, 'limit': 100},
                )
                trades = r.json()
                if not isinstance(trades, list):
                    logger.warning(f'Unexpected response shape: {type(trades)}')
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                new_count = 0
                fetched_at = datetime.now(timezone.utc).isoformat()
                for t in trades:
                    if not isinstance(t, dict):
                        continue
                    tx = t.get('transactionHash', '')
                    if not tx or tx in seen_tx:
                        continue
                    seen_tx.add(tx)
                    new_count += 1

                    ts_epoch = t.get('timestamp')
                    if ts_epoch is None:
                        continue
                    try:
                        ts = datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).isoformat()
                    except (ValueError, TypeError):
                        continue

                    try:
                        price = float(t.get('price', 0))
                        size = float(t.get('size', 0))
                    except (ValueError, TypeError):
                        price, size = 0.0, 0.0

                    writer.writerow([
                        fetched_at, ts, tx, t.get('side', ''),
                        t.get('asset', ''), t.get('conditionId', ''),
                        t.get('outcome', ''), t.get('outcomeIndex', ''),
                        price, size, price * size,
                        t.get('slug', ''), (t.get('title', '') or '')[:200]
                    ])
                fh.flush()

                if new_count > 0:
                    logger.info(f'Logged {new_count} new trades (total seen: {len(seen_tx)})')

                if len(seen_tx) > MAX_SEEN_CACHE:
                    seen_tx = set(list(seen_tx)[-MAX_SEEN_CACHE // 2:])

            except Exception as exc:
                logger.warning(f'Error: {type(exc).__name__}: {exc}')

            await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info('Shutting down')
