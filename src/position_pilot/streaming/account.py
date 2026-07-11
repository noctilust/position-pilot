"""Tastytrade account-streamer protocol messages and decoding."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AccountStreamEvent(BaseModel):
    event_type: str
    data: dict[str, Any]
    timestamp: int | None = None
    sequence: int | None = None


class AccountStreamUnavailable(ConnectionError):
    """The account streamer rejected or could not establish a subscription."""


class AccountStreamerProtocol:
    @staticmethod
    def _oauth_token(access_token: str) -> str:
        return access_token if access_token.startswith("Bearer ") else f"Bearer {access_token}"

    @staticmethod
    def connect(account_numbers: list[str], access_token: str) -> dict:
        return {
            "action": "connect",
            "value": account_numbers,
            "auth-token": AccountStreamerProtocol._oauth_token(access_token),
            "request-id": 2,
        }

    @staticmethod
    def heartbeat(access_token: str) -> dict:
        return {
            "action": "heartbeat",
            "auth-token": AccountStreamerProtocol._oauth_token(access_token),
            "request-id": 1,
        }

    @staticmethod
    def validate_connect_response(message: dict) -> None:
        if message.get("action") != "connect" or message.get("status") != "ok":
            raise AccountStreamUnavailable(str(message.get("message") or "connect rejected"))

    @staticmethod
    def decode(message: dict) -> AccountStreamEvent | None:
        event_type = message.get("type")
        data = message.get("data")
        if not event_type or not isinstance(data, dict):
            return None
        return AccountStreamEvent(
            event_type=str(event_type),
            data=data,
            timestamp=message.get("timestamp"),
            sequence=message.get("ws-sequence") or data.get("sequence"),
        )
