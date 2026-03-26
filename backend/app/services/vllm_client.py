from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class VllmClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            timeout=httpx.Timeout(settings.request_timeout_seconds),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def metrics(self) -> str | None:
        try:
            response = await self._client.get("/metrics")
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            return None

    async def list_models(self) -> httpx.Response:
        return await self._client.get("/v1/models")

    async def create_chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        return await self._client.post("/v1/chat/completions", json=payload)

    async def create_chat_completion_stream(self, payload: dict[str, Any]) -> httpx.Response:
        request = self._client.build_request("POST", "/v1/chat/completions", json=payload)
        return await self._client.send(request, stream=True)

