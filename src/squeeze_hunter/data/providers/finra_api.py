"""FINRA Query API client for consolidated short interest (P6).

Fallback for the public CDN files, which some networks cannot reach (every
request answered 403 from the development machine). Uses the FINRA API
Developer Center's client-credentials flow:

    POST https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token
         ?grant_type=client_credentials          (HTTP Basic: client id / secret)
    POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
         Authorization: Bearer <token>, JSON body with compareFilters /
         dateRangeFilters / limit / offset

Register at https://developer.finra.org and set FINRA_API_CLIENT_ID and
FINRA_API_CLIENT_SECRET. Field names follow the published dataset
definition (settlementDate, symbolCode, currentShortPositionQuantity,
averageDailyVolumeQuantity); if FINRA renames one, `_parse_row` is the
only place to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from squeeze_hunter.data.schema import ShortInterest
from squeeze_hunter.logging_setup import get_logger

log = get_logger("data.finra_api")

TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
BASE_URL = "https://api.finra.org"
DATASET_PATH = "/data/group/otcMarket/name/consolidatedShortInterest"


def api_credentials_from_env() -> tuple[str, str] | None:
    cid = os.environ.get("FINRA_API_CLIENT_ID", "").strip()
    secret = os.environ.get("FINRA_API_CLIENT_SECRET", "").strip()
    return (cid, secret) if cid and secret else None


def _parse_row(row: dict[str, Any]) -> ShortInterest | None:
    try:
        raw_date = str(row["settlementDate"])[:10]
        settlement = datetime.strptime(raw_date, "%Y-%m-%d").date()
        return ShortInterest(
            ticker=str(row["symbolCode"]).strip().upper(),
            settlement_date=settlement,
            si_shares=int(float(row["currentShortPositionQuantity"])),
            si_pct_float=0.0,  # merged from the Yahoo float by the backfill
            avg_daily_volume_20d=int(float(row.get("averageDailyVolumeQuantity", 0) or 0)),
        )
    except (KeyError, ValueError, TypeError) as e:
        log.warning("finra_api_row_unparsable", row=row, err=str(e))
        return None


@dataclass
class FinraApiClient:
    client_id: str
    client_secret: str
    token_url: str = TOKEN_URL
    base_url: str = BASE_URL
    timeout_s: float = 30.0
    page_size: int = 5000
    transport: httpx.AsyncBaseTransport | None = None  # tests inject a MockTransport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport)

    async def _token(self, client: httpx.AsyncClient) -> str:
        r = await client.post(
            self.token_url,
            params={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
        )
        r.raise_for_status()
        token = r.json().get("access_token")
        if not token:
            raise RuntimeError("FINRA API token response carried no access_token")
        return str(token)

    async def fetch_short_interest(
        self, tickers: list[str], since: date | None = None
    ) -> dict[str, list[ShortInterest]]:
        out: dict[str, list[ShortInterest]] = {t: [] for t in tickers}
        start = (since or date(2018, 1, 1)).isoformat()
        end = date.today().isoformat()
        async with self._client() as client:
            token = await self._token(client)
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            for t in tickers:
                offset = 0
                while True:
                    body = {
                        "limit": self.page_size,
                        "offset": offset,
                        "compareFilters": [
                            {"fieldName": "symbolCode", "compareType": "EQUAL", "fieldValue": t}
                        ],
                        "dateRangeFilters": [
                            {"fieldName": "settlementDate", "startDate": start, "endDate": end}
                        ],
                    }
                    r = await client.post(self.base_url + DATASET_PATH, json=body, headers=headers)
                    r.raise_for_status()
                    rows = r.json()
                    if not isinstance(rows, list):
                        raise RuntimeError(f"FINRA API returned a non-list payload for {t}")
                    for row in rows:
                        parsed = _parse_row(row) if isinstance(row, dict) else None
                        if parsed is not None and parsed.ticker == t:
                            out[t].append(parsed)
                    if len(rows) < self.page_size:
                        break
                    offset += self.page_size
        return out
