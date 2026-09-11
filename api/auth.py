from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException

from api.config import Settings, get_settings


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    # No key configured -> auth is opt-in rather than on-by-default, so a local
    # dev setup or a read-mostly public demo doesn't need one just to work.
    # Set API_KEY to lock down the write endpoints this guards.
    if not settings.api_key:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
