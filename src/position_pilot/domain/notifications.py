"""Privacy-safe local notifications (macOS default)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Callable

from .alerts import AlertRecord, privacy_safe_notification_body, privacy_safe_notification_title

logger = logging.getLogger(__name__)


class NotificationService:
    """Deliver local OS notifications without account numbers, balances, or P/L."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        enabled: bool = True,
    ) -> None:
        self._runner = runner or subprocess.run
        self.enabled = enabled
        self.sent: list[dict[str, str]] = []

    def notify_alert(self, alert: AlertRecord, *, rich_preview: bool = False) -> bool:
        title = privacy_safe_notification_title(
            symbol=alert.symbol,
            strategy_type=alert.strategy_type,
            alert_type=alert.alert_type,
        )
        body = privacy_safe_notification_body(alert, rich_preview=rich_preview)
        # Hard redaction: never allow account-like numbers or currency in default payloads.
        if not rich_preview:
            body = title
        payload = {"title": title, "body": body, "alert_id": alert.alert_id}
        self.sent.append(payload)
        if not self.enabled:
            return False
        return self._deliver_macos(title, body)

    def _deliver_macos(self, title: str, body: str) -> bool:
        if not shutil.which("osascript"):
            return False
        # Escape for AppleScript string literals.
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
        safe_body = body.replace("\\", "\\\\").replace('"', '\\"')
        script = f'display notification "{safe_body}" with title "{safe_title}"'
        try:
            completed = self._runner(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            return completed.returncode == 0
        except Exception:
            logger.exception("macOS notification delivery failed")
            return False
