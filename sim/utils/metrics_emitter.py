from __future__ import annotations

import logging
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)


class MetricsEmitter:
    def __init__(self, base_url: str = "http://localhost:4100", timeout_sec: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def send_metric(self, payload: dict[str, Any]) -> None:
        self._post("/metrics", payload)

    def send_status(self, payload: dict[str, Any]) -> None:
        self._post("/status", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        try:
            requests.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("metrics post failed: %s", exc)
