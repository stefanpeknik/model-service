"""Ask EMPAIA to preprocess a slide again, so its preprocessing triggers start jobs.

Usage: request_preprocessing.py SLIDE_ID [WAIT_SECONDS]

Creates a Medical Data Service preprocessing request for the slide; the
Workbench Daemon then starts a job of every app whose trigger matches the
slide's tissue and stain, other apps' included. model-service's own app is
the prostate-heatmap analysis's (portal app
`8e76c21e-4ab4-42c5-91ad-c93fda10f57c`, app
`d3560de0-59f2-4cb9-bc9c-aa01888ce0c5`); its jobs reach
`POST /v1/empaia/analyses/prostate-heatmap`. Run it in an API pod, which has
the MDS URL and client credentials in its environment:

    oc exec -i deploy/model-service-api -- python - SLIDE_ID 120 \
        < scripts/request_preprocessing.py

Prints the request's ID. With WAIT_SECONDS it then polls
`GET /v3/preprocessing-requests/<id>` (the daemon looks every 30 s) and prints
the request as the MDS last answered it, which lists the app jobs it started.
"""

import asyncio
import json
import sys

import httpx

from model_service.providers.empaia.credentials import (
    ClientCredentials,
    NoCredentials,
    OutgoingCredentials,
)
from model_service.providers.empaia.settings import EmpaiaSettings


POLL_SECONDS = 10


async def request(slide_id: str, wait_seconds: float) -> None:
    """Create the preprocessing request, print its ID, and optionally its jobs."""
    settings = EmpaiaSettings()
    if settings.mds_url is None:
        sys.exit("EMPAIA_MDS_URL is not set")
    credentials: OutgoingCredentials = NoCredentials()
    if settings.idp_url and settings.client_id and settings.client_secret:
        credentials = ClientCredentials(
            settings.idp_url,
            settings.client_id,
            settings.client_secret.get_secret_value(),
        )
    mds = f"{settings.mds_url.rstrip('/')}/v3"
    async with httpx.AsyncClient(timeout=30) as client:
        authorization = await credentials.authorization(client)
        headers = {"Authorization": authorization} if authorization else {}
        # The MDS accepts, and cannot delete, a request for an unknown slide.
        slide = await client.get(f"{mds}/slides/{slide_id}", headers=headers)
        slide.raise_for_status()
        response = await client.post(
            f"{mds}/preprocessing-requests",
            json={
                "creator_id": "model-service:request_preprocessing",
                # The only creator type the Medical Data Service knows.
                "creator_type": "USER",
                "slide_id": slide_id,
            },
            headers=headers,
        )
        response.raise_for_status()
        request_id = str(response.json()["id"])
        print(request_id, flush=True)
        latest = response.json()
        waited = 0.0
        while waited < wait_seconds and not latest.get("app_jobs"):
            await asyncio.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            polled = await client.get(
                f"{mds}/preprocessing-requests/{request_id}", headers=headers
            )
            polled.raise_for_status()
            latest = polled.json()
        if wait_seconds:
            print(json.dumps(latest, indent=2))


asyncio.run(request(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0))
