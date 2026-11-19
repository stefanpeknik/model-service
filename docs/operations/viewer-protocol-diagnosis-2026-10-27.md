# Viewer protocol error, 2026-10-27

Read-only investigation of `rationai-infra-test-ns` using `oc -n
rationai-infra-test-ns`. The screenshot shows the CaseViewer slide sidebar
failing with `ReferenceError: leav3 is not defined`.

## Confirmed cause and locations

CaseViewer sends a symbolic protocol name to a viewer that expects a
JavaScript expression.

- Deployment `test-apps-case-viewer`, image
  `cerit.io/rationai/production/case-viewer:DEV-181-3`:
  `/app/app/utils/constants.ts:9` sets `protocol: 'leav3'` in
  `DEFAULT_BG_CONFIG`. `/app/app/utils/xOpat.ts:166` copies that configuration
  into the launch payload. The sidebar submits the payload in its
  `visualization` form field.
- The public `NEXT_PUBLIC_XOPAT_URL` is
  `https://testrat.dyn.cloud.e-infra.cz/xopat/`. The ingress routes `/xopat`
  to `test-apps-xopat`. The separate development viewer uses `/v3/xopat`.
- Deployment `test-apps-xopat`, image
  `cerit.io/rationai/production/xopat:2.3.1-5`:
  `/var/www/html/xopat/src/app.js:1195` constructs
  `new Function("path,data", "return " + (selectedImage.protocol || defaultProtocol))`.
  Line 1196 calls it. With the CaseViewer value, its body becomes
  `return leav3`. JavaScript treats this as a variable lookup and throws
  the reported ReferenceError. The stacked background branch performs the
  same evaluation at line 1181.
- The startup catch at `src/app.js:1078` displays the generic viewer error
  shown in the screenshot.

The viewer's served default `image_group_protocol` is a valid expression
that produces `{"type":"leav3","slide":"<slide-id>"}`. Omitting the
background override allows that default to work.

## Existing mitigation and remaining uncertainty

The currently deployed EMPAIA plugin already removes the symbolic override
in `plugins/empaia/empaia.js:1156`, inside `applyJobVisualizations`.
It registers that method on `before-canvas-reload` at line 40. The minified
plugin contains the deletion too.

An isolated replay of the deployed plugin confirms that successful cleanup
removes the error. The screenshot therefore means this browser session
reached evaluation with the override still present. The investigation did
not capture that browser's stack, loaded script contents, or event handlers,
so it does not establish why the mitigation failed to take effect. Stale
browser assets or an absent handler remain possibilities, not findings.
The served script URLs use the version suffix `v=2.2.2`; both source files
inspected have an October 26 modification date.

## Live observations

- All listed pods were running. No namespace warning events were present.
- Recent Workbench logs contained primitive and pixelmap 404 responses.
  No causal connection between those responses and the JavaScript error
  was established.
- No deployments, configuration, workloads, or credentials were changed.

## Local diagnostic replay

Copied the deployed public `src/app.js`, CaseViewer `constants.ts`, and
EMPAIA plugin source to temporary files. A Node replay extracted the real
background constant and the real viewer evaluation statements. It also
called the real plugin method with an empty job response and a stubbed
background-name update. Results:

```text
Original payload: ReferenceError: leav3 is not defined
After deployed plugin cleanup: {"type":"leav3","slide":"slide-id"}
With explicit protocol expression: {"type":"leav3","slide":"slide-id"}
```

This reproduces the protocol incompatibility, not the full authenticated
browser session. No application test suite was run for this read-only
investigation.

## Correction

For the `/xopat/` release viewer, omit the symbolic `background[].protocol`
override at the CaseViewer launch boundary, or translate it to the valid
EMPAIA expression. Apply the same normalization to initial launch and slide
swaps. Keep any behavior needed by the separate v3 viewer explicit.
Validate the correction in the affected browser before declaring the
incident resolved. No correction was deployed during this investigation.
