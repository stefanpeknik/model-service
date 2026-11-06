# Prostate heatmap beside the EMPAIA test instance, 2026-10-29

The first job from the EMPAIA test instance on model-service deployed in its
own namespace ([rationai-test-ns](rationai-test-ns.md)), reading the slide
directly from the instance's slide volume. The test JES's `model-service`
target was pointed at this deployment for it.

## Results

ProstHM preprocessing job (`4b474468`) on the prostate slide `7a072fd9`,
started from the case browser:

| Attempt | Image | Outcome |
| --- | --- | --- |
| Job `5af750e5`, 21:40 | `empaia-20` | `422`: the app had been given a global configuration key `aigopath_viewer` (xOpat display settings), and the heatmap builder rejected keys it doesn't take |
| Job `e4110196`, 21:53 | `empaia-21` | Key ignored with a warning; JES `TERMINATED` 0 at 22:00, about 7 minutes after acceptance |

Job `e4110196` on the App API: configuration, mode, the slide input, a `404`
for the optional `region_of_interest`, one call to the RationAI direct-access
route, no region reads, then the heatmap output, 18 pixelmap tiles (level 6:
12, level 7: 4, level 8: 2), and finalize. The GPU worker started in about 35 seconds; the TensorRT and
Hugging Face caches started empty.

## Found on the way

- JES logs only the status code of a refused job. The facade now logs every
  refused submission with its reason.
- The facade logged an aiohttp `Unclosed connection` to the App API once,
  after the job; a client session is left for the garbage collector somewhere.
