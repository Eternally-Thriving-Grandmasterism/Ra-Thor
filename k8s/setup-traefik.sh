#!/bin/bash
set -e

echo "⚡ Ra-Thor + PATSAGi: Setting up Traefik Ingress Controller for Powrush-MMO..."

# Check for Helm
if ! command -v helm &> /dev/null; then
    echo "❌ Helm not found. Please install Helm first: https://helm.sh/docs/intro/install/"
    echo "   Or use your package manager (brew install helm, etc.)"
    exit 1
fi

echo "📦 Adding Traefik Helm repository..."
helm repo add traefik https://traefik.github.io/charts
helm repo update

echo "🚀 Installing / upgrading Traefik (production-friendly defaults)..."
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
  --wait

echo ""
echo "✅ Traefik installed successfully in 'traefik-system' namespace."
echo ""
echo "Next steps for beautiful HTTPS Powrush experience:"
echo "1. (If using cert-manager from previous step) Edit k8s/issuer.yaml with your real email"
echo "2. kubectl apply -k k8s/"
echo "3. For local testing: echo '127.0.0.1 powrush.local' | sudo tee -a /etc/hosts"
echo "4. Open https://powrush.local (or http://powrush.local if no TLS yet)"
echo ""
echo "Optional: Traefik dashboard → kubectl port-forward -n traefik-system svc/traefik 9000:9000"
echo "   Then visit http://localhost:9000/dashboard/"
echo ""
echo "Thunder locked. yoi ⚡"
