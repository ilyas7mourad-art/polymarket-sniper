"""Configuration loader for the Polymarket sniper bot."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Loads and validates environment configuration.

    All URL vars are required. Wallet vars are optional in dev phase.
    """

    CLOB_URL: str = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")
    DATA_URL: str = os.getenv("POLYMARKET_DATA_URL", "https://data-api.polymarket.com")
    GAMMA_URL: str = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")
    BINANCE_WS_URL: str = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws")
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    _REQUIRED = ["CLOB_URL", "DATA_URL", "GAMMA_URL", "BINANCE_WS_URL", "POLYGON_RPC_URL"]

    def __init__(self) -> None:
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Raise ValueError if any required URL var is empty."""
        for attr in self._REQUIRED:
            value = getattr(self, attr, "")
            if not value:
                raise ValueError(f"Missing required config var: {attr}")


config = Config()

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
