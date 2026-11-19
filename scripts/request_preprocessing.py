"""Ask EMPAIA to preprocess a slide again, so its preprocessing triggers start jobs.

Usage: request_preprocessing.py SLIDE_ID

Creates a Medical Data Service preprocessing request for the slide; the
Workbench Daemon then starts a job of every app whose trigger matches the
slide's tissue and stain, other apps' included. Run it in an API pod, which has
the MDS URL and client credentials in its environment:

    oc exec -i deploy/model-service-api -- python - SLIDE_ID \
        < scripts/request_preprocessing.py

Prints the request's ID. `GET /v3/preprocessing-requests/<id>` lists the jobs
started for it once the daemon has polled (every 30 s).
"""

import asyncio
import sys

import httpx

from model_service.providers.empaia.credentials import (
    ClientCredentials,
    NoCredentials,
    OutgoingCredentials,
)
from model_service.providers.empaia.settings import EmpaiaSettings


async def request(slide_id: str) -> str:
    """Create the preprocessing request and return its ID."""
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
        return str(response.json()["id"])


print(asyncio.run(request(sys.argv[1])))
