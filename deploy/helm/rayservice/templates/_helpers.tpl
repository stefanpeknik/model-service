{{- define "peknik.proxyEnv" -}}
- name: HTTPS_PROXY
  value: http://proxy.muni.cz:3128
- name: HTTP_PROXY
  value: http://proxy.muni.cz:3128
- name: NO_PROXY
  value: .svc.cluster.local,.svc,10.0.0.0/8,147.251.0.0/16,127.0.0.1
{{- end }}
