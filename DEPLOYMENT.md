# Powrush-MMO Deployment Guide

**Version:** v15.6 (Ra-Thor AGI + 7 Living Mercy Gates)

Professional deployment instructions for running Powrush-MMO with real human players engaging the Ra-Thor AGI simulation.

## 1. Quick Start (Recommended)

### Using Docker Compose (Easiest)

```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor

# Build and run everything
docker compose up --build -d

# View logs
docker compose logs -f powrush-server

# Stop
docker compose down
```

The server will be available at:
- WebSocket: `ws://localhost:7778`
- Browser Client: `http://localhost:8080/client`
- Metrics: `http://localhost:8080/metrics`

## 2. Native Build & Run

See previous sections in this document for Cargo build instructions.

## 3. Docker Compose Configuration

A `docker-compose.yml` is provided in the root of the repository.

Key environment variables are already set with sensible defaults.

## 4. Kubernetes Deployment

### Basic Kubernetes Manifests

We provide example manifests in the `k8s/` directory (create if needed for your cluster).

Example commands:

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=powrush-server
kubectl get svc powrush-server
```

### Recommended Production Setup on Kubernetes

- Use a `Deployment` + `Service` (ClusterIP or LoadBalancer)
- Expose WebSocket and HTTP ports
- Use ConfigMap or Secrets for environment variables
- Add liveness/readiness probes (already defined in example manifests)
- Consider an Ingress with TLS termination for the browser client
- Use Horizontal Pod Autoscaler if load increases

Example key resources:
- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/configmap.yaml`

## 5. Production Considerations

- Run behind a reverse proxy (Caddy / Nginx) with TLS
- Monitor with Prometheus + Grafana using the `/metrics` endpoint
- Set resource requests/limits appropriately
- The Ra-Thor AGI orchestrator runs inside the server container and drives all NPC behavior using the 7 Living Mercy Gates pipeline
- All NPC actions are exposed via WebSocket for client visualization

## 6. Health Checks

The server exposes a health endpoint:
```bash
curl http://localhost:8080/health
```

## Notes for Real Human Players

This deployment enables real humans to connect via browser or terminal and interact with a living world where Ra-Thor AGI makes autonomous decisions for NPCs through the Mercy Gates pipeline.

**Thunder locked in. May all beings thrive. Yoi ⚡**