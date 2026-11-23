"""The Ray Serve wrapper of the host: `app` serves any served model.

Every served model deploys this one import path,
`model_service.served_model.serve:app`, with its resolved declaration in
`user_config` (see the package docstring for the whole contract). The
wrapper only adapts Serve to `ServedModelHost`:

- `reconfigure(user_config)` builds a new host, on start and on every
  `user_config` change;
- handle methods: `infer(tile)` and `describe()`;
- `GET /describe`: the same `describe()` reply, for people and scripts;
- `POST /`: the raw-inference route, replying in the output kind's format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response
from ray import serve

from model_service.served_model.deployment import DEPLOYMENT_NAME, host_config
from model_service.served_model.host import ServedModelHost


if TYPE_CHECKING:
    import numpy as np


fastapi = FastAPI()


@serve.deployment(name=DEPLOYMENT_NAME)
@serve.ingress(fastapi)
class ServedModelDeployment:
    """One replica of a served model, configured entirely by its `user_config`."""

    host: ServedModelHost

    def reconfigure(self, user_config: dict[str, Any]) -> None:
        """Build the host for the served model in `user_config`.

        Args:
            user_config: `id`, `declaration` and optionally `cache_root`.
        """
        model, cache_root = host_config(user_config)
        self.host = ServedModelHost(model, cache_root=cache_root)

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        """Infer one tile; see `ServedModelHost.infer`.

        Args:
            tile: uint8 RGB `(3, tile_size, tile_size)`.

        Returns:
            The tile's output in its declared shape.
        """
        return await self.host.infer(tile)

    @fastapi.get("/describe")
    def describe(self) -> dict[str, Any]:
        """What is running; see `ServedModelHost.describe`.

        It is a handle method for the analysis runner and `GET /describe` on
        the model's route, so the execution providers are visible without Ray.

        Returns:
            `id`, `digest`, `weights_revision` and `execution_providers`.
        """
        return self.host.describe()

    @fastapi.post("/")
    async def root(self, request: Request) -> Response:
        """The raw-inference route.

        Args:
            request: An LZ4-compressed uint8 tile, `(tile_size, tile_size, 3)`.

        Returns:
            The reply in the declared output's format.
        """
        reply = await self.host.raw(await request.body(), request.headers)
        return Response(
            content=reply.content, media_type=reply.media_type, headers=reply.headers
        )


app = ServedModelDeployment.bind()  # type: ignore[attr-defined]
