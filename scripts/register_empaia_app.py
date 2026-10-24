"""Register an EMPAIA app in a marketplace-service mock from its app file.

The app file (`deploy/empaia/apps/<app>.yaml`) gives the builder, the model and
the EMPAIA metadata. The EAD is generated from it, and the app's image becomes
`http::<target>/builders/<builder>/models/<model>`, so JES hands its jobs to the
model-service configured as `<target>` in `JES_HTTP_TARGETS`.

The portal app's IDs are derived from the EAD namespace, so registering the
same app again replaces it. Prints the app ID that jobs are created for.

A marketplace that requires authentication takes the bearer token from the
`EMPAIA_TOKEN` environment variable (e.g. copied from the Workbench).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from generate_ead import generate_ead, load_yaml


def stable_id(kind: str, namespace: str) -> str:
    """A UUID derived from the namespace, shaped as version 4 as the marketplace requires."""
    return str(UUID(bytes=uuid5(NAMESPACE_URL, f"{kind}:{namespace}").bytes, version=4))


def portal_app(
    ead: dict[str, Any], image: str, organization_id: str
) -> dict[str, Any]:
    namespace = ead["namespace"]
    portal_app_id = stable_id("empaia-portal-app", namespace)
    app_id = stable_id("empaia-app", namespace)
    version = namespace[namespace.rfind(".") - 2 :]
    now = int(time.time())
    common = {
        "portal_app_id": portal_app_id,
        "creator_id": organization_id,
        "created_at": now,
    }
    return {
        "id": portal_app_id,
        "organization_id": organization_id,
        "status": "LISTED",
        "creator_id": organization_id,
        "created_at": now,
        "updated_at": now,
        "active_app_views": {
            "v3": {
                **common,
                "organization_id": organization_id,
                "id": stable_id("empaia-app-view", namespace),
                "version": version,
                "api_version": "v3",
                "status": "APPROVED",
                "non_functional": False,
                "research_only": True,
                "details": {
                    "name": ead["name"],
                    "marketplace_url": "http://url.to/store",
                    "description": [{"lang": "EN", "text": ead["description"]}],
                },
                "media": {"peek": [], "banner": [], "workflow": [], "manual": []},
                "tags": {
                    "tissues": [],
                    "stains": [],
                    "indications": [],
                    "analysis": [],
                    "clearances": [],
                },
                "review_comment": "registered by model-service",
                "reviewer_id": organization_id,
                "reviewed_at": now,
                "app": {
                    **common,
                    "id": app_id,
                    "ead": ead,
                    "registry_image_url": image,
                    "app_ui_url": None,
                    "version": version,
                    "status": "APPROVED",
                    "has_frontend": False,
                    "updated_at": now,
                },
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--marketplace-url", required=True)
    parser.add_argument("--target", default="model-service")
    parser.add_argument(
        "--organization-id", default="2c7c2b5e-8d4a-4f5e-9b1a-0d2f6c3e7a10"
    )
    args = parser.parse_args()
    app = load_yaml(args.app)
    ead = generate_ead(app)
    image = f"http::{args.target}/builders/{app['builder']}/models/{app['model']}"
    body = portal_app(ead, image, args.organization_id)
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("EMPAIA_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{args.marketplace_url.rstrip('/')}/v1/custom-mock/portal-apps",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except urllib.error.HTTPError as error:
        sys.exit(f"registration failed: HTTP {error.code}: {error.read().decode()}")
    print(body["active_app_views"]["v3"]["app"]["id"])


if __name__ == "__main__":
    main()
