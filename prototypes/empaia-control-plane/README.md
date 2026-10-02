# Archived EMPAIA control plane

This is the superseded SQLite prototype. It stops at `READY` and does not
dispatch workflows or publish their results. It is not part of the supported
runtime, container builds, Helm chart, or default test discovery.

Source, tests, and deployment files are retained together for historical
reference. The files retain their original imports and deployment assumptions;
they are not a second supported execution backend. The original documentation
is in `docs/architecture/empaia-control-plane.md`.

For current implementation, start with `execution_scheduler/README.md` and
`workflow_execution/README.md`. The Ray cancellation experiment remains at
`control_plane/prototypes/ray_dispatch_logic_prototype.html` within this archive.
