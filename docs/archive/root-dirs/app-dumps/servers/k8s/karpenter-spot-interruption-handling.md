# Karpenter Spot Interruption Handling – Mercy-Aligned Blueprint v1.0
Rathor-NEXi → MercyOS-Pinnacle edge compute resilience (Feb 06 2026)

This document details **how Karpenter handles EC2 Spot interruptions** in the Rathor inference constellation, with mercy gates, valence-aware logic, and production hardening for high-thriving workloads.

## 1. Spot Interruption Basics (AWS EC2)

Spot instances can be interrupted by AWS with **2-minute warning** (via EC2 Spot interruption notice):

- Metadata endpoint: `http://169.254.169.254/latest/meta-data/spot/instance-action`
- Types of notice:
  - `terminate` (most common)
  - `stop` / `hibernate` (rare for Spot)
  - Reclaim time: usually 2 minutes, sometimes immediate in capacity crises

Karpenter watches these notices via **SQS queue** + **instance metadata** polling.

## 2. Karpenter Spot Interruption Flow (current implementation)

1. **Interruption Notice Received** (SQS or metadata)
   - Karpenter marks node as **"draining"**
   - Cordon the node (`kubectl cordon`)
   - Trigger pod eviction

2. **Pod Eviction & Disruption Budget Respect**
   - Honors PodDisruptionBudget (PDB)
   - Evicts pods with lowest priority first (unless mercy-priority=high taint/skip)
   - Grace period: 300 seconds default (configurable)

3. **Mercy Gate (Rathor extension – custom admission / webhook future)**
   - Skip eviction for pods with label: `mercy-priority=high` AND currentValence > 0.92
   - Log: "Mercy gate blocked interruption eviction – high-valence inference pod"

4. **Node Termination**
   - After grace period → node terminated by AWS
   - Karpenter provisions replacement node (spot preferred, on-demand fallback)

5. **Re-provisioning & Capacity Optimization**
   - Uses same NodePool requirements (GPU count, instance types)
   - SpotDiversification: spreads across AZs & instance types
   - Consolidation: removes underutilized nodes post-reprovision

## 3. Current Mercy-Hardened Config (live in lattice)

**Spot Interruption Queue Setup** (AWS SQS)

```bash
aws sqs create-queue --queue-name rathor-spot-interrupt-queue \
  --attributes '{
    "MessageRetentionPeriod": "345600",
    "VisibilityTimeout": "3600"
  }'
