"""Register an EMPAIA app in a marketplace-service mock from its app file.

The app file (`deploy/empaia/apps/<app>.yaml`) gives the builder, the model and
the EMPAIA metadata. The EAD is generated from it, and the app's image becomes
`http::<target>/builders/<builder>/models/<model>`, so JES hands its jobs to the
model-service configured as `<target>` in `JES_HTTP_TARGETS`.

The portal app's IDs are derived from the EAD namespace, so registering the
same app again replaces it. Prints the app ID that jobs are created for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from generate_ead import generate_ead, load_yaml


def portal_app(
    ead: dict[str, Any], image: str, organization_id: str
) -> dict[str, Any]:
    namespace = ead["namespace"]
    portal_app_id = str(uuid5(NAMESPACE_URL, f"empaia-portal-app:{namespace}"))
    app_id = str(uuid5(NAMESPACE_URL, f"empaia-app:{namespace}"))
    version = namespace[namespace.rfind(".") - 2 :]
    now = int(time.time())
    common = {
        "portal_app_id": portal_app_id,
        "organization_id": organization_id,
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
                "id": str(uuid5(NAMESPACE_URL, f"empaia-app-view:{namespace}")),
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
        "--organization-id", default="00000000-0000-0000-0000-000000000000"
    )
    args = parser.parse_args()
    app = load_yaml(args.app)
    ead = generate_ead(app)
    image = f"http::{args.target}/builders/{app['builder']}/models/{app['model']}"
    body = portal_app(ead, image, args.organization_id)
    request = urllib.request.Request(
        f"{args.marketplace_url.rstrip('/')}/v1/custom-mock/portal-apps",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status >= 300:
            sys.exit(f"registration failed: HTTP {response.status}")
    print(body["active_app_views"]["v3"]["app"]["id"])


if __name__ == "__main__":
    main()
