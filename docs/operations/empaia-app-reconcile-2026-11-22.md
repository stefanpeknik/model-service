# EMPAIA app reconcile: delete, not unlist, 2026-11-22

Registration at API start now ends by retiring the apps model-service owns
that no deployed analysis has any more (`registration.py`, `retire_absent`).
This note records how it retires them and what to check on the test instance.
Nothing here ran against a live installation.

## Decision: delete

The marketplace mock can delete a portal app, so retiring deletes it:
`DELETE /v1/custom-mock/portal-apps/{portal_app_id}`. A `404` counts as done,
because another API replica may have deleted it first. Unlisting (posting the
app again with `status` other than `LISTED`) is not used.

Evidence:

- EMPAIA's mock source (`empaia-upstream/marketplace-service-mock`, 0.2.42,
  `api/v1/routes_mps.py`) has `DELETE /custom-mock/portal-apps/{portal_app_id}`
  and `DELETE /custom-mock/portal-apps/by_app/{app_id}`, both `404` for an
  unknown ID, and its tests cover them (`test_delete_portal_app_by_*`).
- The test instance runs `marketplace-service-mock:0.2.28-1`
  ([2026-10-25 note](empaia-test-instance-2026-10-25.md)). Its OpenAPI document
  (`https://testrat.dyn.cloud.e-infra.cz/api/mps/v1/openapi.json`), read in an
  earlier session on 2026-07-24, listed both `DELETE` routes and
  `PUT /customer/portal-apps/query`. No git history of the mock is available
  locally to date the routes otherwise.
- The mock lists apps through `PUT /v1/customer/portal-apps/query` with an
  `organization-id` header; it ignores the query body and returns every portal
  app. Deleting leaves the app's global configuration in the mock's
  configuration store, which nothing reads without the app.

## What model-service owns

A portal app is model-service's when one of its app views has a namespace
starting with `org.empaia.rationai.` **and** an image starting with
`http::<jesTarget>/` (`model-service` on the test instance).

The namespace prefix alone would be too wide. RationAI's container apps use
the same prefix (`org.empaia.rationai.gigapath_embed_score`, `.episeg`,
`.multi_model`, `.prostate_wsi_score` in `rationai-empaia-jobs` and
`empaia-job-template`) and have registry images instead. The image condition
keeps them, and keeps apps another model-service deployment registered under a
different JES target in a shared marketplace.

Apps this rule retires on the test instance, if they are still listed:

| App | Namespace | Image | App ID / portal app ID |
| --- | --- | --- | --- |
| Builder-era prostate heatmap | `org.empaia.rationai.heatmap_prostate_classifier_1.v3.0` | `http::model-service/builders/heatmap/models/prostate-classifier-1` | `71e38adc-fba7-467b-9a04-d21c98c1cdcc` / `a77c0b92-15ca-41d0-bf92-d64bc5b85d58` |
| ConstHMPre (validation app posted by hand on 2026-10-25) | `org.empaia.rationai.constant_heatmap_preprocessing.v3.0` | `http::model-service/builders/heatmap/models/constant-classifier` | derived the same way |

The owner allowed deleting model-service's old apps and triggers there.
The app that stays is `prostate-heatmap`: namespace
`org.empaia.rationai.prostate_heatmap.v3.0`, app `d3560de0-59f2-4cb9-bc9c-aa01888ce0c5`,
portal app `8e76c21e-4ab4-42c5-91ad-c93fda10f57c`.

## Order at startup

1. Post every exposed analysis's app and its global configuration.
2. Align the preprocessing triggers (Medical Data Service): model-service's
   own triggers of a retired app go here, before the app.
3. List the marketplace's apps and delete the owned ones no analysis has.

Any failure retries the whole sequence every minute, logged as a warning and
as an error from the third failure on. Triggers someone else created on a
retired app are left alone, as before.

## For the coordinator to verify live

- The API log at startup (JSON lines, logger
  `model_service.providers.empaia.registration`) has
  `registered EMPAIA app d3560de0-… (standalone, preprocessing) of analysis prostate-heatmap`,
  then `retired EMPAIA portal app a77c0b92-… (org.empaia.rationai.heatmap_prostate_classifier_1.v3.0): no deployed analysis has it`
  on one pod (a line per other retired app, such as ConstHMPre). The pod
  whose delete found the app already gone logs nothing for it, and later
  restarts log no retire lines.
- No `EMPAIA app registration failed` warning. The one risk the source can't
  settle: the test mock uses an LS-AAI integration, and the customer query
  route might reject JES's service-client token although the custom-mock
  routes accept it. A `HTTP 401/403 from …/v1/customer/portal-apps/query`
  warning means that; apps and triggers are already in place by then, only
  retiring waits.
- `PUT /v1/customer/portal-apps/query` (header `organization-id`) afterwards
  lists the prostate-heatmap portal app and every non-owned app as before,
  without `a77c0b92-…`.
- Preprocessing triggers (`GET /v3/preprocessing-triggers` on the MDS): one
  `model-service:model-service` trigger, PROSTATE / H_AND_E, on portal app
  `8e76c21e-…`; none on `a77c0b92-…`.
- The case browser offers Prostate Heatmap once, not twice.
