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
rotating JSON files; the rotation settings bound those. The event-stats dumps
would otherwise flood stdout every minute.
*/}}
{{- define "model-service.rayLogEnv" -}}
- name: RAY_LOG_TO_STDERR
  value: "1"
- name: RAY_SERVE_LOG_TO_STDERR
  value: "1"
- name: RAY_event_stats_print_interval_ms
  value: "-1"
{{ include "model-service.rayLogRotationEnv" . }}
{{- end -}}

{{- define "model-service.rayLogRotationEnv" -}}
- name: RAY_ROTATION_MAX_BYTES
  value: "10485760"
- name: RAY_ROTATION_BACKUP_COUNT
  value: "1"
{{- end -}}
