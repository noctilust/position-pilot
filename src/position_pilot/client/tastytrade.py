"""Tastytrade API client for account and position data."""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from ..models.position import Account, Greeks, Position, PositionType
from ..models.transaction import Transaction, TransactionType, Order, OrderStatus
from ..cache import Cache

# Load .env file from project root
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env")

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

        # Initialize cache with 10-minute TTL
        cache_dir = Path.home() / ".cache" / "position-pilot"
        self._cache = Cache(ttl_seconds=600, cache_dir=cache_dir)

        if not self._enabled:
            logger.warning("Tastytrade credentials not configured")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def clear_cache(self) -> None:
        """Clear all cached market data."""
        self._cache.clear()
        logger.info("Market data cache cleared")

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

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> Optional[dict]:
        """Fetch IV rank and other metrics for a symbol."""
        cache_key = f"metrics:{symbol}"

        # Check cache unless force refresh
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for market metrics: {symbol}")
                return cached

        # Fetch from API
        data = self._get("/market-metrics", params={"symbols": symbol})
        if not data:
            return None

        items = data.get("data", {}).get("items", [])
        if not items:
            return None

        m = items[0]
        result = {
            "iv_rank": self._float(m.get("implied-volatility-index-rank"), mult=100),
            "iv_percentile": self._float(m.get("implied-volatility-percentile"), mult=100),
            "implied_volatility": self._float(m.get("implied-volatility-index")),
            "beta": self._float(m.get("beta")),
            "liquidity_rating": m.get("liquidity-rating"),
        }

        # Store in cache
        self._cache.set(cache_key, result)
        logger.debug(f"Cached market metrics: {symbol}")

        return result

    def get_quote(self, symbol: str, force_refresh: bool = False) -> Optional[dict]:
        """Get current quote for a symbol."""
        cache_key = f"quote:{symbol}"

        # Check cache unless force refresh
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for quote: {symbol}")
                return cached

        # Fetch from API
        data = self._get("/market-data", params={"symbols": symbol})
        if not data:
            return None

        items = data.get("data", {}).get("items", [])
        if not items:
            return None

        q = items[0]
        result = {
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

        # Store in cache
        self._cache.set(cache_key, result)
        logger.debug(f"Cached quote: {symbol}")

        return result

    def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict[str, dict]:
        """Get quotes for multiple symbols in a single API call.

        Returns a dictionary mapping symbol to quote data.
        """
        if not symbols:
            return {}

        quotes = {}
        symbols_to_fetch = []

        # Check cache for each symbol unless force refresh
        if not force_refresh:
            for symbol in symbols:
                cache_key = f"quote:{symbol}"
                cached = self._cache.get(cache_key)
                if cached is not None:
                    quotes[symbol] = cached
                    logger.debug(f"Cache hit for quote: {symbol}")
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = symbols

        # Fetch uncached symbols from API
        if symbols_to_fetch:
            symbols_param = ",".join(symbols_to_fetch)
            data = self._get("/market-data", params={"symbols": symbols_param})
            if data:
                for item in data.get("data", {}).get("items", []):
                    symbol = item.get("symbol", "")
                    if symbol:
                        quote_data = {
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
                        quotes[symbol] = quote_data

                        # Store in cache
                        cache_key = f"quote:{symbol}"
                        self._cache.set(cache_key, quote_data)
                        logger.debug(f"Cached quote: {symbol}")

        return quotes

    def enrich_position_greeks(self, position: Position, force_refresh: bool = False) -> Position:
        """Fetch and attach current Greeks to a position."""
        if not position.is_option:
            return position

        quote = self.get_quote(position.symbol, force_refresh=force_refresh)
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

    def enrich_positions_greeks_batch(self, positions: list[Position], force_refresh: bool = False) -> list[Position]:
        """Fetch and attach Greeks to multiple option positions in a single API call.

        Much more efficient than calling enrich_position_greeks for each position.
        """
        # Filter option positions and collect their symbols
        option_symbols = [p.symbol for p in positions if p.is_option]

        if not option_symbols:
            return positions

        # Fetch all quotes in one batch
        quotes = self.get_quotes_batch(option_symbols, force_refresh=force_refresh)

        # Collect unique underlying symbols for pricing
        underlying_symbols = list(set(p.underlying_symbol for p in positions if p.is_option))
        underlying_quotes = self.get_quotes_batch(underlying_symbols, force_refresh=force_refresh) if underlying_symbols else {}

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

                # Add underlying price for extrinsic value calculation
                if pos.underlying_symbol in underlying_quotes:
                    underlying_quote = underlying_quotes[pos.underlying_symbol]
                    positions[i].underlying_price = underlying_quote.get("mark") or underlying_quote.get("last")

        return positions

    def get_transactions(
        self,
        account_number: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 250,
        force_refresh: bool = False
    ) -> list[Transaction]:
        """Fetch transaction history for an account.

        Args:
            account_number: The account number
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of transactions to fetch (default: 250)
            force_refresh: If True, bypass cache

        Returns:
            List of Transaction objects
        """
        cache_key = f"transactions:{account_number}"

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for transactions: {account_number}")
                # Return all cached transactions (filtering by date in Python)
                transactions = [Transaction(**t) for t in cached.get("transactions", [])]
                if start_date or end_date:
                    transactions = self._filter_transactions_by_date(transactions, start_date, end_date)
                return transactions

        # Fetch from API
        params = {"limit": limit}
        if start_date:
            params["start-date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end-date"] = end_date.strftime("%Y-%m-%d")

        data = self._get(f"/accounts/{account_number}/transactions", params=params)
        if not data:
            return []

        transactions = []
        items = data.get("data", {}).get("items", [])

        for item in items:
            tx = self._parse_transaction(item, account_number)
            if tx:
                transactions.append(tx)

        # Store in cache (store all transactions without date filtering)
        cache_data = {
            "transactions": [t.model_dump(mode="json", exclude_none=True) for t in transactions],
            "cached_at": datetime.now().isoformat()
        }
        self._cache.set(cache_key, cache_data)
        logger.debug(f"Cached {len(transactions)} transactions for {account_number}")

        return transactions

    def get_orders(
        self,
        account_number: str,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        limit: int = 100
    ) -> list[Order]:
        """Fetch order history for an account.

        Args:
            account_number: The account number
            status: Optional status filter (e.g., "filled", "open")
            start_date: Optional start date filter
            limit: Maximum number of orders to fetch (default: 100)

        Returns:
            List of Order objects
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if start_date:
            params["start-date"] = start_date.strftime("%Y-%m-%d")

        data = self._get(f"/accounts/{account_number}/orders", params=params)
        if not data:
            return []

        orders = []
        items = data.get("data", {}).get("items", [])

        for item in items:
            order = self._parse_order(item, account_number)
            if order:
                orders.append(order)

        return orders

    def _filter_transactions_by_date(
        self,
        transactions: list[Transaction],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> list[Transaction]:
        """Filter transactions by date range."""
        filtered = transactions
        if start_date:
            filtered = [t for t in filtered if t.transaction_date >= start_date]
        if end_date:
            filtered = [t for t in filtered if t.transaction_date <= end_date]
        return filtered

    def _parse_transaction(self, item: dict, account_number: str) -> Optional[Transaction]:
        """Parse a transaction from API response."""
        try:
            # Parse transaction type
            tx_type_str = item.get("type", "").lower().replace("-", "").replace("_", "")
            tx_type = TransactionType.ORDER_FILL  # Default

            for t in TransactionType:
                if t.value in tx_type_str or tx_type_str in t.value:
                    tx_type = t
                    break

            # Parse transaction date
            date_str = item.get("transaction-date", item.get("date", ""))
            if date_str:
                try:
                    # Tastytrade returns ISO format with microseconds
                    if "." in date_str:
                        tx_date = datetime.strptime(date_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                    else:
                        tx_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    tx_date = datetime.now()
            else:
                tx_date = datetime.now()

            return Transaction(
                transaction_id=item.get("id", ""),
                transaction_type=tx_type,
                transaction_date=tx_date,
                description=item.get("description", ""),
                amount=self._float(item.get("value", item.get("amount", 0))),
                symbol=item.get("symbol"),
                quantity=self._float(item.get("quantity")),
                price=self._float(item.get("price")),
                commission=self._float(item.get("commission", item.get("fees"))),
                order_id=item.get("order-id"),
                account_number=account_number
            )
        except Exception as e:
            logger.error(f"Error parsing transaction: {e}")
            return None

    def _parse_order(self, item: dict, account_number: str) -> Optional[Order]:
        """Parse an order from API response."""
        try:
            # Parse order status
            status_str = item.get("status", "").lower()
            order_status = OrderStatus.RECEIVED
            for s in OrderStatus:
                if s.value in status_str:
                    order_status = s
                    break

            # Parse dates
            created_str = item.get("created-at", "")
            updated_str = item.get("updated-at", "")

            try:
                if created_str:
                    created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                else:
                    created_at = datetime.now()
            except ValueError:
                created_at = datetime.now()

            try:
                if updated_str:
                    updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                else:
                    updated_at = datetime.now()
            except ValueError:
                updated_at = datetime.now()

            # Get leg details (for multi-leg orders)
            legs = item.get("legs", [])
            if legs:
                # Use first leg for symbol/action
                leg = legs[0]
                symbol = leg.get("symbol", "")
                action = leg.get("action", "")
            else:
                symbol = item.get("symbol", "")
                action = item.get("action", "")

            return Order(
                order_id=item.get("id", ""),
                account_number=account_number,
                symbol=symbol,
                action=action,
                quantity=self._float(item.get("quantity", item.get("total-quantity"))),
                order_type=item.get("type", "market"),
                status=order_status,
                created_at=created_at,
                updated_at=updated_at,
                filled_quantity=self._float(item.get("filled-quantity")),
                average_fill_price=self._float(item.get("average-fill-price")),
                commissions=self._float(item.get("-commissions", 0), mult=-1),  # Tastytrade returns negative
                underlying_symbol=item.get("underlying-symbol")
            )
        except Exception as e:
            logger.error(f"Error parsing order: {e}")
            return None

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
