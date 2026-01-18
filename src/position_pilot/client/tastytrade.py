"""Tastytrade API client for account and position data."""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from ..models.position import Account, Greeks, Position, PositionType

load_dotenv()

logger = logging.getLogger(__name__)

TASTYTRADE_API_URL = "https://api.tastyworks.com"


class TastytradeClient:
    """Client for Tastytrade API - accounts, positions, and market data."""

    def __init__(self):
        self.client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "")
        self.refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._enabled = bool(self.client_secret and self.refresh_token)

        if not self._enabled:
            logger.warning("Tastytrade credentials not configured")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _ensure_token(self) -> bool:
        """Ensure we have a valid access token."""
        if not self._enabled:
            return False

        if self._access_token and self._token_expiry and datetime.now() < self._token_expiry:
            return True

        try:
            response = httpx.post(
                f"{TASTYTRADE_API_URL}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                },
                timeout=10.0,
            )
            response.raise_for_status()

            data = response.json()
            self._access_token = data.get("access_token")
            expires_in = data.get("expires_in", 900)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)

            logger.info("Tastytrade token refreshed")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"Token refresh failed: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make authenticated GET request."""
        if not self._ensure_token():
            return None

        try:
            response = httpx.get(
                f"{TASTYTRADE_API_URL}{endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {self._access_token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API error {endpoint}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Request error {endpoint}: {e}")
            return None

    def get_accounts(self) -> list[Account]:
        """Fetch all accounts for the authenticated user."""
        data = self._get("/customers/me/accounts")
        if not data:
            return []

        accounts = []
        for item in data.get("data", {}).get("items", []):
            acc = item.get("account", {})
            accounts.append(
                Account(
                    account_number=acc.get("account-number", ""),
                    account_type=acc.get("account-type-name", ""),
                    nickname=acc.get("nickname"),
                    day_trade_count=self._get_day_trade_count(acc.get("day-trader-status")),
                )
            )

        return accounts

    def get_account_balances(self, account_number: str) -> Optional[dict]:
        """Fetch balance details for an account."""
        data = self._get(f"/accounts/{account_number}/balances")
        if not data:
            return None

        bal = data.get("data", {})
        return {
            "net_liquidating_value": self._float(bal.get("net-liquidating-value")),
            "cash_balance": self._float(bal.get("cash-balance")),
            "buying_power": self._float(bal.get("derivative-buying-power")),
            "maintenance_excess": self._float(bal.get("maintenance-excess")),
            "day_trading_buying_power": self._float(bal.get("day-trading-buying-power")),
            "pnl_today": self._float(bal.get("pending-cash")),
        }

    def get_positions(self, account_number: str) -> list[Position]:
        """Fetch all positions for an account."""
        data = self._get(f"/accounts/{account_number}/positions")
        if not data:
            return []

        positions = []
        for item in data.get("data", {}).get("items", []):
            pos = self._parse_position(item)
            if pos:
                positions.append(pos)

        return positions

    def _parse_position(self, item: dict) -> Optional[Position]:
        """Parse a position from API response."""
        try:
            symbol = item.get("symbol", "")
            underlying = item.get("underlying-symbol", symbol)
            instrument_type = item.get("instrument-type", "")
            quantity = int(item.get("quantity", 0))
            quantity_dir = item.get("quantity-direction", "Long")

            # Determine position type
            pos_type = PositionType.EQUITY
            if instrument_type == "Equity Option":
                pos_type = PositionType.EQUITY_OPTION
            elif instrument_type == "Future":
                pos_type = PositionType.FUTURE
            elif instrument_type == "Future Option":
                pos_type = PositionType.FUTURE_OPTION
            elif instrument_type == "Cryptocurrency":
                pos_type = PositionType.CRYPTOCURRENCY

            # Parse option details
            strike = None
            opt_type = None
            exp_date = None
            dte = None

            if pos_type == PositionType.EQUITY_OPTION:
                strike = self._float(item.get("strike-price"))
                opt_type = item.get("option-type", "")[:1].upper()  # "C" or "P"

                exp_str = item.get("expiration-date", "")
                if exp_str:
                    try:
                        exp_date = datetime.strptime(exp_str[:10], "%Y-%m-%d").date()
                        dte = (exp_date - date.today()).days
                    except ValueError:
                        pass

                # Fallback: parse OCC symbol if API didn't provide details
                # OCC format: SYMBOL (6 chars) + YYMMDD + C/P + strike*1000 (8 chars)
                if (not strike or not opt_type or not exp_date) and len(symbol) >= 15:
                    parsed = self._parse_occ_symbol(symbol)
                    if parsed:
                        if not strike:
                            strike = parsed.get("strike")
                        if not opt_type:
                            opt_type = parsed.get("option_type")
                        if not exp_date:
                            exp_date = parsed.get("expiration_date")
                            if exp_date:
                                dte = (exp_date - date.today()).days

            # Pricing
            avg_open = self._float(item.get("average-open-price")) or 0.0
            close = self._float(item.get("close-price")) or 0.0
            mark = self._float(item.get("mark-price"))
            multiplier = int(item.get("multiplier", 100 if pos_type == PositionType.EQUITY_OPTION else 1))

            # P/L calculations
            cost_basis = avg_open * abs(quantity) * multiplier
            market_value = (mark or close) * abs(quantity) * multiplier

            if quantity_dir == "Short":
                unrealized_pnl = cost_basis - market_value
            else:
                unrealized_pnl = market_value - cost_basis

            unrealized_pnl_pct = None
            if cost_basis != 0:
                unrealized_pnl_pct = (unrealized_pnl / abs(cost_basis)) * 100

            realized = self._float(item.get("realized-day-gain")) or 0.0

            return Position(
                symbol=symbol,
                underlying_symbol=underlying,
                quantity=quantity,
                quantity_direction=quantity_dir,
                position_type=pos_type,
                strike_price=strike,
                option_type=opt_type,
                expiration_date=exp_date,
                days_to_expiration=dte,
                average_open_price=avg_open,
                close_price=close,
                mark_price=mark,
                cost_basis=cost_basis,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_pct,
                realized_pnl=realized,
                multiplier=multiplier,
            )

        except Exception as e:
            logger.error(f"Error parsing position: {e}")
            return None

    def get_market_metrics(self, symbol: str) -> Optional[dict]:
        """Fetch IV rank and other metrics for a symbol."""
        data = self._get("/market-metrics", params={"symbols": symbol})
        if not data:
            return None

        items = data.get("data", {}).get("items", [])
        if not items:
            return None

        m = items[0]
        return {
            "iv_rank": self._float(m.get("implied-volatility-index-rank"), mult=100),
            "iv_percentile": self._float(m.get("implied-volatility-percentile"), mult=100),
            "implied_volatility": self._float(m.get("implied-volatility-index")),
            "beta": self._float(m.get("beta")),
            "liquidity_rating": m.get("liquidity-rating"),
        }

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Get current quote for a symbol."""
        data = self._get("/market-data", params={"symbols": symbol})
        if not data:
            return None

        items = data.get("data", {}).get("items", [])
        if not items:
            return None

        q = items[0]
        return {
            "bid": self._float(q.get("bid")),
            "ask": self._float(q.get("ask")),
            "mark": self._float(q.get("mark")),
            "last": self._float(q.get("last")),
            "delta": self._float(q.get("delta")),
            "gamma": self._float(q.get("gamma")),
            "theta": self._float(q.get("theta")),
            "vega": self._float(q.get("vega")),
            "implied_volatility": self._float(q.get("volatility")),
        }

    def get_quotes_batch(self, symbols: list[str]) -> dict[str, dict]:
        """Get quotes for multiple symbols in a single API call.

        Returns a dictionary mapping symbol to quote data.
        """
        if not symbols:
            return {}

        # Tastytrade API accepts comma-separated symbols
        symbols_param = ",".join(symbols)
        data = self._get("/market-data", params={"symbols": symbols_param})
        if not data:
            return {}

        quotes = {}
        for item in data.get("data", {}).get("items", []):
            symbol = item.get("symbol", "")
            if symbol:
                quotes[symbol] = {
                    "bid": self._float(item.get("bid")),
                    "ask": self._float(item.get("ask")),
                    "mark": self._float(item.get("mark")),
                    "last": self._float(item.get("last")),
                    "delta": self._float(item.get("delta")),
                    "gamma": self._float(item.get("gamma")),
                    "theta": self._float(item.get("theta")),
                    "vega": self._float(item.get("vega")),
                    "implied_volatility": self._float(item.get("volatility")),
                }

        return quotes

    def enrich_position_greeks(self, position: Position) -> Position:
        """Fetch and attach current Greeks to a position."""
        if not position.is_option:
            return position

        quote = self.get_quote(position.symbol)
        if quote:
            position.greeks = Greeks(
                delta=quote.get("delta"),
                gamma=quote.get("gamma"),
                theta=quote.get("theta"),
                vega=quote.get("vega"),
                implied_volatility=quote.get("implied_volatility"),
            )
            if quote.get("mark"):
                position.mark_price = quote["mark"]

        return position

    def enrich_positions_greeks_batch(self, positions: list[Position]) -> list[Position]:
        """Fetch and attach Greeks to multiple option positions in a single API call.

        Much more efficient than calling enrich_position_greeks for each position.
        """
        # Filter option positions and collect their symbols
        option_symbols = [p.symbol for p in positions if p.is_option]

        if not option_symbols:
            return positions

        # Fetch all quotes in one batch
        quotes = self.get_quotes_batch(option_symbols)

        # Update positions with quote data
        for i, pos in enumerate(positions):
            if pos.is_option and pos.symbol in quotes:
                quote = quotes[pos.symbol]
                positions[i].greeks = Greeks(
                    delta=quote.get("delta"),
                    gamma=quote.get("gamma"),
                    theta=quote.get("theta"),
                    vega=quote.get("vega"),
                    implied_volatility=quote.get("implied_volatility"),
                )
                if quote.get("mark"):
                    positions[i].mark_price = quote["mark"]

        return positions

    def _get_day_trade_count(self, status: Any) -> int:
        """Extract day trade count from status field."""
        if isinstance(status, dict):
            return status.get("day-trade-count", 0)
        return 0

    def _float(self, val: Any, mult: float = 1.0) -> Optional[float]:
        """Safely convert to float."""
        if val is None:
            return None
        try:
            return float(val) * mult
        except (TypeError, ValueError):
            return None

    def _parse_occ_symbol(self, symbol: str) -> Optional[dict]:
        """
        Parse OCC option symbol format.

        Format: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
        Example: 'AMD   260220P00200000' -> AMD Feb 20 2026 $200 Put
        """
        try:
            # Remove any spaces and ensure minimum length
            symbol = symbol.replace(" ", "")
            if len(symbol) < 15:
                return None

            # Extract components from the end (more reliable)
            strike_str = symbol[-8:]  # Last 8 chars: strike * 1000
            opt_type = symbol[-9]  # C or P
            exp_str = symbol[-15:-9]  # YYMMDD

            # Parse strike (divide by 1000)
            strike = int(strike_str) / 1000.0

            # Parse expiration date
            exp_date = datetime.strptime(exp_str, "%y%m%d").date()

            # Validate option type
            if opt_type not in ("C", "P"):
                return None

            return {
                "strike": strike,
                "option_type": opt_type,
                "expiration_date": exp_date,
            }
        except (ValueError, IndexError):
            return None


# Global singleton
_client: Optional[TastytradeClient] = None


def get_client() -> TastytradeClient:
    """Get or create the global client instance."""
    global _client
    if _client is None:
        _client = TastytradeClient()
    return _client
