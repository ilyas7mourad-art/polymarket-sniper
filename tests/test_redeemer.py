"""Unit tests for src/redeemer.py — all offline."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.redeemer import REDEMPTION_INTERVAL_SECONDS, Redeemer


def _make_redeemer(tmp_path: Path) -> Redeemer:
    """Create a Redeemer with a mocked ClobClient and a temp data dir."""
    with patch("src.redeemer.ClobClient"), \
         patch("src.redeemer.config") as mock_cfg:
        mock_cfg.CLOB_HOST = "https://clob.polymarket.com"
        mock_cfg.CHAIN_ID = 137
        mock_cfg.WALLET_PRIVATE_KEY = ""
        mock_cfg.WALLET_FUNDER = ""
        mock_cfg.WALLET_ADDRESS = "0xtest"
        mock_cfg.DATA_DIR = str(tmp_path)
        redeemer = Redeemer()
    return redeemer


def _make_http_client(positions: list) -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = positions
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def test_constant() -> None:
    assert REDEMPTION_INTERVAL_SECONDS == 300


def test_no_positions(tmp_path: Path) -> None:
    """Empty position list → no redemptions, no crash."""
    redeemer = _make_redeemer(tmp_path)
    mock_http = _make_http_client([])

    with patch("src.redeemer.httpx.AsyncClient", return_value=mock_http), \
         patch("src.redeemer.config") as mock_cfg:
        mock_cfg.WALLET_ADDRESS = "0xtest"
        asyncio.run(redeemer._check_and_redeem())

    assert len(redeemer._redeemed_conditions) == 0


def test_skips_already_redeemed(tmp_path: Path) -> None:
    """Condition already in redeemed set → not retried."""
    redeemer = _make_redeemer(tmp_path)
    redeemer._redeemed_conditions.add("cond-111")

    positions = [{"conditionId": "cond-111", "redeemable": True, "currentValue": "10.0"}]
    mock_http = _make_http_client(positions)

    with patch("src.redeemer.httpx.AsyncClient", return_value=mock_http), \
         patch("src.redeemer.config") as mock_cfg:
        mock_cfg.WALLET_ADDRESS = "0xtest"
        asyncio.run(redeemer._check_and_redeem())

    # Still only one entry (not double-added)
    assert redeemer._redeemed_conditions == {"cond-111"}


def test_skips_non_redeemable(tmp_path: Path) -> None:
    """Position with redeemable=False → not queued."""
    redeemer = _make_redeemer(tmp_path)
    positions = [{"conditionId": "cond-222", "redeemable": False, "currentValue": "5.0"}]
    mock_http = _make_http_client(positions)

    with patch("src.redeemer.httpx.AsyncClient", return_value=mock_http), \
         patch("src.redeemer.config") as mock_cfg:
        mock_cfg.WALLET_ADDRESS = "0xtest"
        asyncio.run(redeemer._check_and_redeem())

    assert len(redeemer._redeemed_conditions) == 0


def test_redemption_failure_does_not_add_condition(tmp_path: Path) -> None:
    """_redeem_one raising NotImplementedError → condition NOT added to redeemed set."""
    redeemer = _make_redeemer(tmp_path)
    positions = [{"conditionId": "cond-333", "redeemable": True, "currentValue": "8.0"}]
    mock_http = _make_http_client(positions)

    with patch("src.redeemer.httpx.AsyncClient", return_value=mock_http), \
         patch("src.redeemer.config") as mock_cfg, \
         patch.object(redeemer, "_redeem_one", side_effect=NotImplementedError("no redeem method")):
        mock_cfg.WALLET_ADDRESS = "0xtest"
        asyncio.run(redeemer._check_and_redeem())

    assert "cond-333" not in redeemer._redeemed_conditions


def test_redemption_success_adds_condition(tmp_path: Path) -> None:
    """_redeem_one succeeding → condition added to redeemed set."""
    redeemer = _make_redeemer(tmp_path)
    positions = [{"conditionId": "cond-444", "redeemable": True, "currentValue": "12.0"}]
    mock_http = _make_http_client(positions)

    with patch("src.redeemer.httpx.AsyncClient", return_value=mock_http), \
         patch("src.redeemer.config") as mock_cfg, \
         patch.object(redeemer, "_redeem_one", return_value=None):
        mock_cfg.WALLET_ADDRESS = "0xtest"
        asyncio.run(redeemer._check_and_redeem())

    assert "cond-444" in redeemer._redeemed_conditions


def test_api_error_does_not_crash(tmp_path: Path) -> None:
    """httpx failure → logged and swallowed, no exception raised."""
    redeemer = _make_redeemer(tmp_path)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.redeemer.httpx.AsyncClient", return_value=mock_client), \
         patch("src.redeemer.config") as mock_cfg:
        mock_cfg.WALLET_ADDRESS = "0xtest"
        asyncio.run(redeemer._check_and_redeem())  # must not raise

    assert len(redeemer._redeemed_conditions) == 0


def test_redeem_one_raises_not_implemented(tmp_path: Path) -> None:
    """_redeem_one raises NotImplementedError when client lacks both redeem methods."""
    redeemer = _make_redeemer(tmp_path)
    # Replace client with a plain object that has no redeem attrs
    redeemer._client = object()

    with pytest.raises(NotImplementedError, match="py-clob-client"):
        redeemer._redeem_one("cond-555")


def test_state_persisted_and_loaded(tmp_path: Path) -> None:
    """Redeemed conditions survive a save/load cycle."""
    redeemer = _make_redeemer(tmp_path)
    redeemer._redeemed_conditions.add("cond-aaa")
    redeemer._redeemed_conditions.add("cond-bbb")
    redeemer._save_state()

    with patch("src.redeemer.ClobClient"), \
         patch("src.redeemer.config") as mock_cfg:
        mock_cfg.DATA_DIR = str(tmp_path)
        redeemer2 = Redeemer()

    assert "cond-aaa" in redeemer2._redeemed_conditions
    assert "cond-bbb" in redeemer2._redeemed_conditions
