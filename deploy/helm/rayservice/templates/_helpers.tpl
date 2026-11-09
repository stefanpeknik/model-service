{{/* Extra DNS search domains, for installations that hand out short App Service names. */}}
{{- define "model-service.empaiaDns" -}}
{{- with .Values.empaia.dnsSearches }}
dnsConfig:
  searches: {{ toJson . }}
{{- end }}
{{- end }}
