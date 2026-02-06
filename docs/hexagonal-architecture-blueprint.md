# Hexagonal Architecture (Ports & Adapters) – Rathor-NEXi Blueprint v1.0
(February 06 2026 – MercyOS-Pinnacle core architectural pattern)

This living document defines how **Hexagonal Architecture** (Ports & Adapters pattern by Alistair Cockburn) is applied across the Rathor-NEXi monorepo — isolating the **pure domain core** (valence logic, mercy gates, gesture semantics, swarm rules, negotiation invariants) from all external concerns (UI, APIs, persistence, ML runtimes, WebGPU/WebNN, IndexedDB, ElectricSQL, Yjs, etc.).

Goal: **maximum domain sovereignty**, **testability**, **technology-agnostic core**, **easy adapter swapping**, **valence-enforced boundaries**.

## 1. Core Hexagonal Principles in Rathor-NEXi

- **Domain Core** (inside hexagon) — pure business logic, no framework, no I/O, no external dependency
  - Aggregates, Entities, Value Objects, Domain Services, Domain Events
  - Mercy gates, valence projection, thriving invariants
  - No knowledge of React, WebLLM, IndexedDB, ONNX, TensorRT, etc.

- **Ports** (hexagon boundary interfaces)
  - Inbound Ports (driven): use-case interfaces (Application Services)
  - Outbound Ports (driving): repository interfaces, notification ports, ML inference ports, sync ports

- **Adapters** (outside hexagon) — implement ports, depend on external tech
  - Primary Adapters (drive the app): React UI, CLI, gRPC server, FastAPI proxy
  - Secondary Adapters (driven by app): IndexedDB repo, WebNN inference, ElectricSQL sync, Yjs CRDT

## 2. Rathor-NEXi Hexagonal Layering (2026 current)
