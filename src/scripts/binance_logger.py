"""Log Binance BTC and ETH trade prices to CSV via aggTrade WebSocket.

Streams aggTrade for both pairs concurrently. One row per trade.
Rotates CSV daily. Reconnects on disconnect.
"""
import asyncio
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import websockets

DATA_DIR = Path('/home/mma/polymarket-sniper/data')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

WS_URL = 'wss://stream.binance.com:9443/stream?streams=btcusdt@aggTrade/ethusdt@aggTrade'
HEADER = ['timestamp_utc', 'symbol', 'price', 'qty', 'is_buyer_maker']


def get_csv_path() -> Path:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return DATA_DIR / f'binance_{today}.csv'


async def run() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_path = None
    writer = None
    fh = None

    while True:
        try:
            logger.info(f'Connecting to {WS_URL}')
            async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=10) as ws:
                logger.info('Connected')
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    data = msg.get('data', {})

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

                    ts_ms = data.get('E')
                    if ts_ms is None:
                        continue
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
                    symbol = data.get('s', '')
                    price = data.get('p', '')
                    qty = data.get('q', '')
                    is_buyer_maker = data.get('m', False)
                    writer.writerow([ts, symbol, price, qty, is_buyer_maker])
                    fh.flush()
        except Exception as exc:
            logger.warning(f'WS error: {type(exc).__name__}: {exc}, reconnecting in 5s')
            await asyncio.sleep(5)


if __name__ == '__main__':
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info('Shutting down')
