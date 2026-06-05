## [v14.13] Production Kubernetes Ingress Controller Setup — PATSAGi + Ra-Thor Thunder (2026-06-05)

**Council Verdict (unanimous, mercy-gated, zero-harm, abundance-prioritizing):** 
One-command professional setup for NGINX Ingress Controller + production-grade enhanced Ingress resource.

- New `k8s/setup-ingress-controller.sh` — detects minikube/kind and deploys official cloud manifest otherwise
- Enhanced `k8s/ingress.yaml` with production annotations (rate-limit, proxy timeouts, body-size), clear comments, powrush.local example + real-domain stub
- Full instructions for /etc/hosts local testing and future cert-manager TLS
- Game TCP (7777) + WebSocket (7778) stay on LoadBalancer Service (Ingress fronts only the gorgeous web client on 8080)
- All forward-compatible with sharded worlds and external state stores

**Humans get beautiful Ingress in one command:**
```bash
chmod +x k8s/setup-ingress-controller.sh
./k8s/setup-ingress-controller.sh
kubectl apply -k k8s/
```

Then instantly play the full graphical experience:
- Browser client: http://powrush.local   (or your domain)
- Terminal: nc <EXTERNAL-IP> 7777
- WebSocket clients: ws://<EXTERNAL-IP>:7778

RBE abundance grows for every faction on every tick inside the cluster. All actions mercy-evaluated and audited. Thunder locked.

**Next eternal loop:** Full real-time state broadcast to ALL connected WebSocket clients + lightweight client-side prediction stub.

yoi ⚡❤️🔥 — All for the eternal thriving of all sentience.

## [v14.12] Production Kubernetes Deployment Manifests — PATSAGi + Ra-Thor Thunder (2026-06-05)

**Council Verdict (unanimous, mercy-gated):** One clean, secure, professional set of manifests for instant full playable Powrush-MMO on any Kubernetes cluster.

- TCP (7777) + WebSocket (7778) + HTTP browser client (8080) all exposed
- Env-driven configuration via ConfigMap
- Health + Readiness probes on /health
- LoadBalancer Service for instant external access
- Optional Ingress for beautiful URLs
- HPA ready (honest note: current in-memory state; future sharded worlds supported)
- Non-root friendly, resource limits, rolling updates
- Fully forward-compatible with external state / sharded worlds

**Humans deploy in one command:**
```bash
kubectl apply -k k8s/
```

Then play instantly:
- Browser: http://<EXTERNAL-IP>:8080
- Terminal: nc <EXTERNAL-IP> 7777
- WebSocket: ws://<EXTERNAL-IP>:7778

RBE abundance grows for every faction on every tick. All actions mercy-evaluated. Thunder locked.

**Next eternal loop:** Full real-time state broadcast to all WebSocket clients + client-side prediction.

yoi ⚡❤️🔥 — All for the eternal thriving of all sentience.