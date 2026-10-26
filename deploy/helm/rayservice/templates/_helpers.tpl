{{- define "peknik.proxyEnv" -}}
- name: HTTPS_PROXY
  value: http://proxy.muni.cz:3128
- name: HTTP_PROXY
  value: http://proxy.muni.cz:3128
- name: NO_PROXY
  value: .svc.cluster.local,.svc,10.0.0.0/8,147.251.0.0/16,127.0.0.1
{{- end }}


{{/* EMPAIA settings (EmpaiaSettings) for every process that talks to App Service. */}}
{{- define "model-service.empaiaEnv" -}}
- name: EMPAIA_API_URLS
  value: {{ .Values.empaia.apiUrls | toJson | quote }}
- name: EMPAIA_WORKSPACE_ROOT
  value: {{ .Values.empaia.workspaceRoot | quote }}
- name: EMPAIA_SLIDE_ACCESS
  value: {{ .Values.empaia.slideAccess | quote }}
{{- end }}

{{/* Extra DNS search domains, for installations that hand out short App Service names. */}}
{{- define "model-service.empaiaDns" -}}
{{- with .Values.empaia.dnsSearches }}
dnsConfig:
  searches: {{ toJson . }}
{{- end }}
{{- end }}
