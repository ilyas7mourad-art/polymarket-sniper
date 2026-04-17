# Polymarket CLOB Orderbook — WebSocket Reference

Discovered during PR #4 implementation (2026-04-18). Verified against live API.

## WebSocket URL

```
wss://ws-subscriptions-clob.polymarket.com/ws/market
```

No authentication required for the market channel (read-only).

## Subscription message

Send immediately after connecting. Use `asset_id` (CLOB token IDs), not condition IDs.

```json
{
  "type": "market",
  "assets_ids": ["<up_token_id>", "<down_token_id>", "..."]
}
```

You can subscribe to multiple token IDs in one message. You can also send additional
subscription messages on the same connection to add more tokens later (additive, not
replacing).

## Initial response — full book snapshot

On subscription, the server responds with a **list** of book snapshots, one per subscribed
`asset_id`:

```json
[
  {
    "event_type": "book",
    "market": "0x2580e3a9...",
    "asset_id": "31968822433894845242...",
    "timestamp": "1776468965039",
    "hash": "08e8132c...",
    "bids": [
      {"price": "0.65", "size": "8743.02"},
      {"price": "0.64", "size": "1511.13"}
    ],
    "asks": [
      {"price": "0.66", "size": "8946.67"},
      {"price": "0.67", "size": "200.00"}
    ],
    "tick_size": "0.01",
    "last_trade_price": "0.65"
  },
  {
    "event_type": "book",
    "asset_id": "<down_token_id>",
    ...
  }
]
```

`bids` are ordered highest-price-first. `asks` are ordered lowest-price-first.
So `bids[0].price` = best bid, `asks[0].price` = best ask.

## Incremental updates — price_change

Subsequent messages arrive as a **list** with `event_type: "price_change"`:

```json
[
  {
    "event_type": "price_change",
    "market": "0x...",
    "asset_id": "...",
    "timestamp": "...",
    "changes": [
      {"price": "0.65", "side": "BUY", "size": "0"},
      {"price": "0.64", "side": "BUY", "size": "150.0"}
    ]
  }
]
```

`side` is `"BUY"` (bid) or `"SELL"` (ask). `size: "0"` removes the level.

## Heartbeat

The server sends periodic `PING` frames (not JSON). The `websockets` library handles
`PONG` automatically. No explicit ping/pong logic needed in application code.

## Bids/asks ordering convention

- **Bids**: highest price first → `bids[0]` is best bid
- **Asks**: lowest price first → `asks[0]` is best ask

## Auth

- **Market channel** (`/ws/market`): no auth
- **User channel** (`/ws/user`): requires API credentials (not used here)

## REST snapshot endpoint

If WebSocket is unavailable, fall back to:

```
GET https://clob.polymarket.com/book?token_id=<asset_id>
```

Response has same `bids` / `asks` structure. Confirmed 200 OK, no auth required.

## Notes

- For markets that are hours from resolution, the spread is typically 0.01/0.99
  (essentially no real orders). The interesting data is in the last ~90 seconds.
- Confirmed live: subscribing to 40 token IDs (20 markets × 2 sides) in one
  message works fine.
