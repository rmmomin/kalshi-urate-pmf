"""Thin wrapper around the Kalshi HTTP API with RSA-PSS per-request signing."""

from __future__ import annotations

import base64
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils
from dotenv import load_dotenv

from src.utils import KALSHI_API_BASE

load_dotenv()


class KalshiClient:
    """Authenticated client for the Kalshi v2 REST API."""

    def __init__(
        self,
        key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str = KALSHI_API_BASE,
    ):
        self.key_id = key_id or os.environ["KALSHI_KEY_ID"]
        pk_path = private_key_path or os.environ["KALSHI_PRIVATE_KEY_PATH"]
        with open(pk_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def _sign_request(self, method: str, path: str) -> dict:
        """Generate per-request auth headers.

        Message: "{timestamp_ms}{METHOD}{full_path}" (full path WITHOUT query params).
        Signing: RSA-PSS with SHA256, salt_length = DIGEST_LENGTH.
        """
        timestamp_ms = str(int(time.time() * 1000))
        # The signed path must include the /trade-api/v2 prefix
        full_path = path if path.startswith("/trade-api") else f"/trade-api/v2{path}"
        message = f"{timestamp_ms}{method.upper()}{full_path}"
        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._sign_request("GET", path)
        resp = self.session.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Market endpoints
    # ------------------------------------------------------------------
    def get_market(self, ticker: str) -> dict:
        """GET /markets/{ticker} → market metadata."""
        return self._get(f"/markets/{ticker}")

    def get_markets(self, event_ticker: str, limit: int = 100) -> list[dict]:
        """GET /markets?event_ticker=... → list of markets in an event."""
        data = self._get("/markets", params={"event_ticker": event_ticker, "limit": limit})
        return data.get("markets", [])

    def get_orderbook(self, ticker: str) -> dict:
        """GET /markets/{ticker}/orderbook → current orderbook snapshot."""
        return self._get(f"/markets/{ticker}/orderbook")

    # ------------------------------------------------------------------
    # High-level: strike surface
    # ------------------------------------------------------------------
    def get_strike_surface(
        self,
        event_ticker: str,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Fetch all contracts for an event, extract strikes and prices.

        Returns a DataFrame with columns:
            ticker, strike, yes_bid, yes_ask, mid, last_price, yes_price, volume
        where *yes_price* follows the preference chain: mid → last → 0.
        """
        markets = self.get_markets(event_ticker)
        if not markets:
            raise ValueError(f"No markets found for event_ticker={event_ticker!r}")

        rows = []
        for mkt in markets:
            ticker = mkt.get("ticker", "")
            subtitle = mkt.get("subtitle", "") or mkt.get("title", "")
            # Parse strike: handles "Above 4.3%" or just "4.3%"
            m = re.search(r"([\d.]+)%", subtitle)
            if not m:
                continue
            strike = float(m.group(1))

            # Orderbook
            ob = self.get_orderbook(ticker)
            orderbook = ob.get("orderbook", {})
            yes_bids = orderbook.get("yes", [])  # [[price, qty], ...]
            no_bids = orderbook.get("no", [])

            # Best yes bid = highest price someone will pay for yes
            best_yes_bid = max((p for p, q in yes_bids), default=None) if yes_bids else None
            # Best no bid = highest no price → tightest yes_ask = 100 - max(no_bid)
            best_no_bid = max((p for p, q in no_bids), default=None) if no_bids else None
            best_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None

            mid = None
            if best_yes_bid is not None and best_yes_ask is not None:
                mid = (best_yes_bid + best_yes_ask) / 2.0

            last_price = mkt.get("last_price", None)
            volume = mkt.get("volume", 0)

            # Price preference: mid > last > None
            yes_price = mid if mid is not None else last_price
            if yes_price is None:
                yes_price = 0.0

            rows.append({
                "ticker": ticker,
                "strike": strike,
                "yes_bid": best_yes_bid,
                "yes_ask": best_yes_ask,
                "mid": mid,
                "last_price": last_price,
                "yes_price": yes_price,
                "volume": volume,
            })

        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        asof = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        df.insert(0, "asof_timestamp", asof)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

        return df
