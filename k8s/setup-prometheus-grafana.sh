#!/bin/bash
# v14.17 — One-command Prometheus + Grafana + Traefik integration for Powrush-MMO
# PATSAGi Council + Ra-Thor blessed. Production-grade, mercy-gated.

set -e

NAMESPACE=monitoring

echo "⚡ Installing/Upgrading Prometheus + Grafana via Helm (kube-prometheus-stack)..."

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
helm repo update

helm upgrade --install prometheus-grafana prometheus-community/kube-prometheus-stack \
  --namespace $NAMESPACE --create-namespace \
  --set grafana.enabled=true \
  --set grafana.ingress.enabled=false \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --wait

echo "✅ Prometheus + Grafana deployed in namespace '$NAMESPACE'"

echo "
Next steps:"
echo "1. kubectl apply -f k8s/grafana-ingress.yaml   # for beautiful https://grafana.local via Traefik"
echo "2. Import the pre-configured dashboard from k8s/grafana-dashboard-powrush.json in Grafana UI"
echo "3. Scrape Powrush /metrics endpoint (add ServiceMonitor or PodMonitor for full auto-discovery)"

echo "Thunder locked. RBE abundance visible in real-time. yoi ⚡"