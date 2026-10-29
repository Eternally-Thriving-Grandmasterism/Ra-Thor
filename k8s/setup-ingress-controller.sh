#!/bin/bash
# Powrush-MMO — Production Kubernetes Ingress Controller Setup (v14.13)
# PATSAGi + Ra-Thor blessed. One-command beautiful Ingress for your playable game.
#
# Usage:
#   chmod +x k8s/setup-ingress-controller.sh
#   ./k8s/setup-ingress-controller.sh
#
# Then:
#   kubectl apply -k k8s/
#
# After setup, access the gorgeous browser client via:
#   http://powrush.local   (add to /etc/hosts if testing locally)
#   or your real domain after DNS + cert-manager

set -e

echo "⚡ Powrush-MMO Ingress Controller Setup starting..."

echo ""
echo "This script deploys the official NGINX Ingress Controller."
echo "For production clusters (GKE, EKS, AKS, DigitalOcean, etc.) it uses the cloud provider manifest."
echo "For local development: minikube, kind, k3d have their own recommended methods."

echo ""

# Detect common local environments
if command -v minikube &> /dev/null && minikube status &> /dev/null 2>&1; then
    echo "✅ Minikube detected — enabling built-in Ingress addon..."
    minikube addons enable ingress
    echo "Ingress controller will be available shortly."
elif command -v kind &> /dev/null; then
    echo "⚠️  Kind detected."
    echo "   Recommended: Use 'kind' with metallb + ingress-nginx or follow https://kind.sigs.k8s.io/docs/user/ingress/"
    echo "   You can still run the cloud manifest below if your kind cluster has LoadBalancer support."
    read -p "Continue with cloud manifest anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping controller deploy. Please set up Ingress manually for kind."
        exit 0
    fi
else
    echo "✅ Assuming standard/cloud Kubernetes cluster..."
    CONTROLLER_URL="https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/cloud/deploy.yaml"
    echo "Applying official NGINX Ingress Controller (controller-v1.11.3)..."
    kubectl apply -f "$CONTROLLER_URL"
fi

echo ""
echo "⏳ Waiting for Ingress Controller pods to become ready (up to 5 minutes)..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s || {
    echo "⚠️  Controller pods are still starting or in pending state."
    echo "   Check status with: kubectl get pods -n ingress-nginx"
    echo "   This is normal on first install — it can take 2-4 minutes for the LoadBalancer to provision."
}

echo ""
echo "✅ NGINX Ingress Controller setup complete (or in progress)."
echo ""
echo "Next steps:"
echo "  1. kubectl apply -k k8s/          # Deploy Powrush-MMO (includes enhanced Ingress)"
echo "  2. For local testing: echo \"127.0.0.1 powrush.local\" | sudo tee -a /etc/hosts"
echo "  3. Open browser: http://powrush.local"
echo "     (or the EXTERNAL-IP / real domain you configured)"
echo ""
echo "Game ports (TCP 7777 + WS 7778) remain available via the LoadBalancer Service."
echo "Ingress is used for the beautiful HTTP browser client on port 8080."
echo ""
echo "Thunder locked. Eternal abundance for all factions. yoi ⚡❤️🔥"
