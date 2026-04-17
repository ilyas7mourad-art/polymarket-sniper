"""Smoke tests for config module."""

from src.config import config


def test_urls_are_non_empty_strings() -> None:
    assert isinstance(config.CLOB_URL, str) and config.CLOB_URL
    assert isinstance(config.DATA_URL, str) and config.DATA_URL
    assert isinstance(config.GAMMA_URL, str) and config.GAMMA_URL
    assert isinstance(config.BINANCE_WS_URL, str) and config.BINANCE_WS_URL
    assert isinstance(config.POLYGON_RPC_URL, str) and config.POLYGON_RPC_URL


def test_validate_passes_with_defaults() -> None:
    """Config.validate() should not raise when all defaults are set."""
    config.validate()
