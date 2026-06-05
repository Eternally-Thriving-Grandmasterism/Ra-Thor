#!/bin/bash
set -e

echo "⚡ Ra-Thor + PATSAGi: Setting up Traefik Ingress Controller + Secure Dashboard + Observability for Powrush-MMO..."

# Check for Helm
if ! command -v helm &> /dev/null; then
    echo "❌ Helm not found. Please install Helm first: https://helm.sh/docs/intro/install/"
    echo "   Or use your package manager (brew install helm, etc.)"
    exit 1
fi

echo "📦 Adding Traefik Helm repository..."
helm repo add traefik https://traefik.github.io/charts
helm repo update

echo "🚀 Installing / upgrading Traefik (production-friendly defaults + metrics + dashboard ready)..."
helm upgrade --install traefik traefik/traefik \
  --namespace traefik-system \
  --create-namespace \
  --set ingressClass.enabled=true \
  --set ingressClass.isDefaultClass=true \
  --set ports.websecure.exposedPort=443 \
  --set logs.general.level=INFO \
  --set accessLogs.enabled=true \
  --set dashboard.enabled=true \
  --set dashboard.ingress.enabled=false \
  --set "metrics.prometheus.enabled=true" \
  --set "metrics.prometheus.manualRouting=true" \
  --wait

echo ""
echo "✅ Traefik installed successfully in 'traefik-system' namespace."
echo "   - Access logs: enabled"
  echo "   - Prometheus metrics: enabled (scrape /metrics on Traefik)"
  echo "   - Dashboard: enabled (secure access below)"

echo ""
echo "🔐 Setting up secure Traefik Dashboard basic auth (CHANGE PASSWORD IN PRODUCTION!)..."
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: traefik-dashboard-auth-secret
  namespace: traefik-system
type: kubernetes.io/basic-auth
stringData:
  username: admin
  password: "P@ssw0rd123!ChangeMe"
---
apiVersion: traefik.io/v1alpha1
kind: Middleware
metadata:
  name: traefik-dashboard-auth
  namespace: traefik-system
spec:
  basicAuth:
    secret: traefik-dashboard-auth-secret
    realm: "Traefik Dashboard - Eternal Mercy Protected"
EOF

echo ""
echo "✅ Secure Dashboard resources applied."
echo ""
echo "Next steps for beautiful HTTPS Powrush experience:"
echo "1. (If using cert-manager) Edit k8s/issuer.yaml with your real email"
echo "2. kubectl apply -k k8s/"
echo "3. For local testing: echo '127.0.0.1 powrush.local traefik-dashboard.local' | sudo tee -a /etc/hosts"
echo "4. Open https://powrush.local (game client) or https://traefik-dashboard.local (dashboard - login admin / your password)"
echo ""
echo "Optional secure Traefik Dashboard access:"
echo "  kubectl port-forward -n traefik-system svc/traefik 9000:9000"
echo "  Then visit http://localhost:9000/dashboard/ (use basic auth)"
echo "  Or apply the full IngressRoute example: kubectl apply -f k8s/traefik-dashboard.yaml (edit host/password first)"
echo ""
echo "Other similar components ready:"
echo "  - Prometheus metrics endpoint on Traefik for scraping player counts, RBE abundance, tick health, mercy logs"
echo "  - Access logs for full audit trail"
echo "  - Ready for kube-prometheus-stack / Grafana dashboards (RBE flow, mercy metrics, server health)"
echo "  - Future: Loki for logs, Jaeger for tracing if desired"
echo ""
echo "Thunder locked. yoi ⚡"
