# Powrush-MMO SurrealDB + TiKV 3-Node Cluster Deployment

**PATSAGi Council Approved — Production-grade local clustering for testing horizontal scaling & high availability.**

## Quick Start
```bash
cd powrush/deploy/surrealdb-3node-tikv
docker compose up -d
```

- surreal1 → http://localhost:8001
- surreal2 → http://localhost:8002
- surreal3 → http://localhost:8003

Connect from code with any node or load balancer:
```rust
let config = SurrealConfig {
    endpoint: "ws://localhost:8001".to_string(),
    cluster_nodes: vec![
        "ws://localhost:8002".into(),
        "ws://localhost:8003".into(),
    ],
    ...
};
```

## Architecture
- 3x TiKV + 1x PD (Placement Driver) for distributed KV storage
- 3x SurrealDB nodes connected to the TiKV cluster
- Full horizontal scaling & replication
- ACID transactions across nodes

## Production Notes
- Use this as template for Kubernetes (StatefulSets + TiKV Operator recommended)
- For SurrealDB Cloud Scale (GA 2026 Q2): replace with managed endpoint
- Enable incremental backups once roadmap feature lands
- Monitor via SurrealDB metrics + TiKV dashboard

## Next (after this cluster spins)
1. Run `init_schema()` from any node
2. Test save/load with the strong-typed methods in v15.3
3. Wire into simulation_orchestrator via persistence_tick_system

All aligned with AG-SML, TOLC 8, and mercy principles.
