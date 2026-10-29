# Powrush-MMO Kubernetes Deployment (SurrealDB + TiKV)

**Production-grade Kubernetes manifests** for the 3-node SurrealDB + TiKV cluster.

## Quick Deploy
```bash
kubectl apply -f namespace.yaml
kubectl apply -f pd.yaml
kubectl apply -f tikv-statefulset.yaml
kubectl apply -f surrealdb-deployment.yaml
```

## Access from Rust Code
Use the internal Service DNS or expose via LoadBalancer/Ingress:

```rust
let config = SurrealConfig {
    endpoint: "ws://surrealdb.powrush-mmo.svc.cluster.local:8000".to_string(),
    cluster_nodes: vec![],
    namespace: "powrush".to_string(),
    database: "mmo_v15".to_string(),
    ...
};
```

## Scaling
- TiKV: Increase `replicas` in StatefulSet (add more storage as needed)
- SurrealDB: Scale the Deployment (add more replicas for query throughput)
- For very large scale: Use TiKV Operator + SurrealDB Cloud Scale

## Production Recommendations
- Add PersistentVolumeClaims with appropriate storage class
- Enable PodDisruptionBudgets
- Monitor with Prometheus + TiKV dashboard
- Use NetworkPolicies for security

All manifests follow best practices for stateful workloads and are ready for production Powrush-MMO clusters.
