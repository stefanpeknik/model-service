{{- define "peknik.proxyEnv" -}}
- name: HTTPS_PROXY
  value: http://proxy.muni.cz:3128
- name: HTTP_PROXY
  value: http://proxy.muni.cz:3128
- name: NO_PROXY
  value: .svc.cluster.local,.svc,10.0.0.0/8,147.251.0.0/16,127.0.0.1
{{- end }}


{{/*
EMPAIA settings (EmpaiaSettings) for every process that talks to App Service.
The EMPAIA helpers render nothing while empaia.apiUrls is empty.
*/}}
{{- define "model-service.empaiaEnv" -}}
{{- if .Values.empaia.apiUrls -}}
- name: EMPAIA_API_URLS
  value: {{ .Values.empaia.apiUrls | toJson | quote }}
- name: EMPAIA_WORKSPACE_ROOT
  value: {{ .Values.empaia.workspaceRoot | quote }}
{{- end }}
{{- end }}

{{/*
The installation's slide volume, where direct access links slides from. It is
read-only at the volume, not only the mount, so kubelet never re-owns the
EMPAIA files to this pod's fsGroup.
*/}}
{{- define "model-service.slideVolumeMount" -}}
{{- with .Values.empaia.slideVolume }}
{{- if and $.Values.empaia.apiUrls .existingClaim }}
- name: empaia-slides
  mountPath: {{ required "empaia.slideVolume.mountPath is required with a claim" .mountPath | quote }}
  readOnly: true
{{- end }}
{{- end }}
{{- end }}

{{- define "model-service.slideVolume" -}}
{{- with .Values.empaia.slideVolume }}
{{- if and $.Values.empaia.apiUrls .existingClaim }}
- name: empaia-slides
  persistentVolumeClaim:
    claimName: {{ .existingClaim | quote }}
    readOnly: true
{{- end }}
{{- end }}
{{- end }}

{{/* Extra DNS search domains, for installations that hand out short App Service names. */}}
{{- define "model-service.empaiaDns" -}}
{{- if .Values.empaia.apiUrls }}
{{- with .Values.empaia.dnsSearches }}
dnsConfig:
  searches: {{ toJson . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Ray logs go to the container's stdout/stderr, where the cluster's log collector
picks them up, instead of files under /tmp/ray. Serve still writes its own
rotating JSON files; the rotation settings bound those. The event-stats and
GCS debug-state dumps would otherwise fill stdout every minute; they are put
off to once a day. Not -1: Ray 2.58's GCS then dumps in a tight loop
(gcs_server.cc schedules it without the -1 check the raylet has).
*/}}
{{- define "model-service.rayLogEnv" -}}
- name: RAY_LOG_TO_STDERR
  value: "1"
- name: RAY_SERVE_LOG_TO_STDERR
  value: "1"
- name: RAY_event_stats_print_interval_ms
  value: "86400000"
{{ include "model-service.rayLogRotationEnv" . }}
{{- end -}}

{{- define "model-service.rayLogRotationEnv" -}}
- name: RAY_ROTATION_MAX_BYTES
  value: "10485760"
- name: RAY_ROTATION_BACKUP_COUNT
  value: "1"
{{- end -}}

{{/*
Deep-merge `patch` over `base` the way the catalog loader does
(model_service.catalog.merge): mappings merge key by key, any other value
replaces, null removes the key. Not sprig's mergeOverwrite, which keeps the
base value when the patch says false, 0 or "". Renders JSON.
*/}}
{{- define "model-service.deepMerge" -}}
{{- $merged := deepCopy .base }}
{{- range $key, $value := .patch }}
{{- if kindIs "invalid" $value }}
{{- $_ := unset $merged $key }}
{{- else if and (kindIs "map" $value) (kindIs "map" (get $merged $key)) }}
{{- $_ := set $merged $key (include "model-service.deepMerge" (dict "base" (get $merged $key) "patch" $value) | fromJson) }}
{{- else }}
{{- $_ := set $merged $key $value }}
{{- end }}
{{- end }}
{{- toJson $merged }}
{{- end }}

{{/*
The documents of one catalog folder this release deploys, as JSON
{id: document}: each catalog/<folder>/<id>.yaml (through the chart's `catalog`
symlink to the repo's catalog/) that the values key `key` selects (["*"] or
named IDs), with its catalog/environments/<environment>/<folder>/<id>.yaml
patch merged in. Takes dict "root" ($), "folder" and "key".
*/}}
{{- define "model-service.catalogDocuments" -}}
{{- $root := .root }}
{{- $folder := .folder }}
{{- $selected := index $root.Values .key | default list }}
{{- $env := $root.Values.environment | default "" }}
{{- $documents := dict }}
{{- range $path, $_ := $root.Files.Glob (printf "catalog/%s/*.yaml" $folder) }}
{{- $id := base $path | trimSuffix ".yaml" }}
{{- if or (has "*" $selected) (has $id $selected) }}
{{- $document := $root.Files.Get $path | fromYaml | default dict }}
{{- if hasKey $document "Error" }}{{ fail (printf "%s is not valid YAML: %s" $path $document.Error) }}{{ end }}
{{- $patch := $root.Files.Get (printf "catalog/environments/%s/%s/%s" $env $folder (base $path)) }}
{{- if and $env $patch }}
{{- $document = include "model-service.deepMerge" (dict "base" $document "patch" (fromYaml $patch | default dict)) | fromJson }}
{{- end }}
{{- $_ := set $documents $id $document }}
{{- end }}
{{- end }}
{{- range $path, $_ := $root.Files.Glob (printf "catalog/environments/%s/%s/*.yaml" $env $folder) }}
{{- if not ($root.Files.Get (printf "catalog/%s/%s" $folder (base $path))) }}{{ fail (printf "%s patches a file catalog/%s/ does not have" $path $folder) }}{{ end }}
{{- end }}
{{- range $selected }}
{{- if and (ne . "*") (not (hasKey $documents .)) }}{{ fail (printf "%s names %s, but catalog/%s/%s.yaml does not exist" $.key . $folder .) }}{{ end }}
{{- end }}
{{- toJson $documents }}
{{- end }}

{{/*
The served models this release deploys, as JSON {id: document}: the
catalog/served-models/ files `servedModels` selects, patched for `environment`.
*/}}
{{- define "model-service.servedModels" -}}
{{- $models := include "model-service.catalogDocuments" (dict "root" . "folder" "served-models" "key" "servedModels") | fromJson }}
{{- range $.Values.models }}
{{- if hasKey $models (base .) }}{{ fail (printf "%s is deployed from catalog/served-models/ and from models/%s.yaml; drop it from models:" (base .) .) }}{{ end }}
{{- end }}
{{- toJson $models }}
{{- end }}

{{/*
The analyses this release deploys, as JSON {id: document}: the
catalog/analyses/ files `analyses` selects, patched for `environment`. An
analysis whose `{model: <id>}` step names a served model this release does not
deploy fails the render, naming both.
*/}}
{{- define "model-service.analyses" -}}
{{- $analyses := include "model-service.catalogDocuments" (dict "root" . "folder" "analyses" "key" "analyses") | fromJson }}
{{- $models := include "model-service.servedModels" . | fromJson }}
{{- range $id, $analysis := $analyses }}
{{- range $analysis.steps | default list }}
{{- if and (kindIs "map" .) (hasKey . "model") (not (hasKey $models (toString .model))) }}
{{- fail (printf "analysis %s calls served model %s, which this release does not deploy: add it to servedModels or leave %s out of analyses" $id .model $id) }}
{{- end }}
{{- end }}
{{- end }}
{{- toJson $analyses }}
{{- end }}

{{/*
One served model's Serve application, as
model_service.served_model.deployment.serve_application builds it, plus the
environment's env vars (servedModelSettings.env) as the replica's runtime_env.
servedModelSettings.setupTimeoutSeconds, when set, goes to the application's
runtime_env.config. Takes dict "id", "document" and "settings".
*/}}
{{- define "model-service.servedModelApplication" -}}
{{- $id := .id }}
{{- $model := .document }}
{{- $batch := int64 (required (printf "served model %s needs batching.max_size" $id) (dig "batching" "max_size" nil $model)) }}
{{- $serving := $model.serving | default dict }}
{{- $resources := $model.resources | default dict }}
{{- $memory := required (printf "served model %s needs resources.memory_gib" $id) $resources.memory_gib }}
{{- $actor := dict "num_cpus" ($resources.cpus | default 1) "num_gpus" ($resources.gpus | default 0) "memory" (mulf $memory 1073741824 | int64) }}
{{- with .settings.env }}
{{- $_ := set $actor "runtime_env" (dict "env_vars" .) }}
{{- end }}
{{- $autoscaling := dict "min_replicas" ($serving.min_replicas | default 0) "max_replicas" ($serving.max_replicas | default 4) "target_ongoing_requests" ($serving.target_ongoing_requests | default (max (div $batch 2) 1)) }}
{{- if not (kindIs "invalid" $serving.downscale_delay_s) }}
{{- $_ := set $autoscaling "downscale_delay_s" $serving.downscale_delay_s }}
{{- end }}
{{- $userConfig := dict "id" $id "declaration" $model "cache_root" (required "servedModelSettings.cacheRoot is required" .settings.cacheRoot) }}
{{- $deployment := dict "name" "ServedModel" "max_ongoing_requests" ($serving.max_ongoing_requests | default (mul 2 $batch)) "max_queued_requests" ($serving.max_queued_requests | default (mul 16 $batch)) "autoscaling_config" $autoscaling "ray_actor_options" $actor "user_config" $userConfig }}
{{- $application := dict "name" $id "route_prefix" (printf "/%s" $id) "import_path" "model_service.served_model.serve:app" "deployments" (list $deployment) }}
{{- with .settings.setupTimeoutSeconds }}
{{- $_ := set $application "runtime_env" (dict "config" (dict "setup_timeout_seconds" (int64 .))) }}
{{- end }}
{{- toYaml $application }}
{{- end }}

{{/*
Provider blocks (e.g. `empaia`) of the deployed served models, as JSON
{id: {block: value}}, from provider-blocks/<id>.yaml. Transitional: today's
model files carry them and served-model files don't, until the blocks move
to the analyses' `expose`.
*/}}
{{- define "model-service.providerBlocks" -}}
{{- $models := include "model-service.servedModels" . | fromJson }}
{{- $blocks := dict }}
{{- range $path, $_ := .Files.Glob "provider-blocks/*.yaml" }}
{{- $id := base $path | trimSuffix ".yaml" }}
{{- if hasKey $models $id }}
{{- $_ := set $blocks $id ($.Files.Get $path | fromYaml) }}
{{- end }}
{{- end }}
{{- toJson $blocks }}
{{- end }}
