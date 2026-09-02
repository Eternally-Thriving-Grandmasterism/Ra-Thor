# Karpenter + PodDisruptionBudget – Mercy-Aligned Integration Notes v1.0

Karpenter **honors PodDisruptionBudgets (PDBs)** during consolidation, spot interruption handling, and node expiration.  
Here is how the lattice enforces mercy during disruptions.

## 1. How Karpenter Interacts with PDBs

- **Consolidation** (WhenUnderutilized / WhenEmpty):
  - Karpenter only consolidates nodes if **all pods can be safely evicted** respecting PDBs
  - If a PDB prevents eviction → node is **not** consolidated

- **Spot Interruption / Scheduled Maintenance**:
  - Karpenter drains node respecting PDB minAvailable / maxUnavailable
  - If PDB blocks → Karpenter waits until PDB allows or node is force-terminated by AWS (2-min warning)

- **Expiration** (expireAfter):
  - Same PDB respect logic — no forced eviction if PDB violated

## 2. Mercy Gate Extensions (Current & Planned)

Current (manifest-level):
- minAvailable ensures at least N replicas survive voluntary disruptions
- High-valence inference pods get `mercy-priority=high` label → future admission webhook will skip them

Planned (next strike):
- Custom Admission Webhook:
  - Block voluntary eviction of pods with `mercy-priority=high` AND projected future valence ≥ 0.92
  - Log: "Mercy gate blocked PDB eviction – high-valence inference pod"

## 3. Recommended PDB Settings for Rathor Workloads

Triton Inference Server:
- minAvailable: 2 (for 3–15 replicas)
- maxUnavailable: 1 (alternative)
- Protects against simultaneous model reloads / restarts during peak inference

FastAPI Proxy:
- minAvailable: 1 (for 2–10 replicas)
- Ensures at least one proxy always available for browser/WebSocket clients

## 4. Monitoring & Alerts

Prometheus queries:

```promql
# PDB violations (pods unavailable beyond budget)
sum(pod_disruption_budget_status_current_healthy < pod_disruption_budget_status_desired_healthy) by (namespace, poddisruptionbudget)
