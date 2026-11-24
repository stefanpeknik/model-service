"""The generic served-model host: one Serve deployment class for every served model.

A served model is a file in `catalog/served-models/`; this package serves it.

- `host.py`: `ServedModelHost`, a plain class built from a resolved
  `ServedModel`. It fetches the weights, loads the runtime, and owns batching,
  `infer(tile)`, the raw-inference reply, the input and output guards and
  `describe()`. Tests and tools use it without Serve.
- `serve.py`: the thin Ray Serve wrapper, `app`, with the HTTP routes
  (`POST /`, `GET /describe`) and the handle methods.
- `deployment.py`: `host_config`, which reads the `user_config` Helm renders.

Deployment contract (what Helm renders for each served model, one application
each, in `deploy/helm/templates/_helpers.tpl`):

```yaml
- name: <id>                     # the file name
  route_prefix: /<id>
  import_path: model_service.served_model.serve:app   # the same for every model
  runtime_env: {config: {setup_timeout_seconds: …}}   # the environment's, only when set
  deployments:
    - name: ServedModel
      max_ongoing_requests: <serving.max_ongoing_requests or 2 x batching.max_size>
      max_queued_requests: <serving.max_queued_requests or 16 x batching.max_size>
      autoscaling_config:
        min_replicas: <serving.min_replicas, 0>
        max_replicas: <serving.max_replicas, 4>
        target_ongoing_requests: <serving.target_ongoing_requests or batching.max_size / 2>
        downscale_delay_s: <serving.downscale_delay_s, only when given>
      ray_actor_options:
        num_cpus: <resources.cpus>
        num_gpus: <resources.gpus>
        memory: <resources.memory_gib in bytes>
        runtime_env: {env_vars: …}   # from the environment's values, e.g. MLFLOW_TRACKING_URI
      user_config:
        id: <id>
        declaration: <the served-model document after the environment's patch>
        cache_root: <the environment's cache root; /mnt/cache/trt_cache by default>
```

The host resolves `declaration` with the same loader code as the catalog, so
`describe()["digest"]` equals the catalog's digest. A changed `user_config`
reconfigures the replica in place. Runtime caches (TensorRT engines) go to
`<cache_root>/<id>/<first 12 hex digits of the digest>`.

Only `serve.py` imports Ray; the other modules load in the control plane.
"""
