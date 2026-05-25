# Next.js + Rust (`web-forge`) Integration Guide

This document outlines recommended patterns for integrating a **Next.js** frontend with a **Rust backend** powered by `web-forge`.

## Recommended Architecture

```
Next.js Frontend (UI + tRPC)
        │
        ▼
Rust Backend (Axum + web-forge)
        │
        ├── Orchestration
        ├── WCAG AA Scoring
        ├── Reporting
        └── Observability
```

**Why this split?**
- Next.js excels at frontend, UI/UX, and developer experience.
- Rust + `web-forge` excels at orchestration logic, performance, accessibility scoring, and strong reporting.

## Recommended Tech Stack

| Layer           | Technology                    | Reason |
|-----------------|-------------------------------|--------|
| Frontend        | Next.js 15 (App Router)       | Best DX + performance |
| Type Safety     | tRPC                          | End-to-end type safety |
| Backend         | Rust + Axum                   | Performance + `web-forge` |
| Communication   | REST (JSON) / Connect         | Simple and effective |
| Auth            | JWT                           | Standard and secure |
| Observability   | OpenTelemetry                 | Already supported in `web-forge` |

## Project Structure Recommendation

```bash
web-forge-monorepo/
├── backend/                 # Rust + web-forge
│   └── src/
│       ├── routes/
│       │   ├── orchestrate.rs
│       │   └── reports.rs
│       └── main.rs
│
├── frontend/                # Next.js
│   ├── app/
│   ├── components/
│   └── lib/trpc/
│
└── docs/
```

## Key Endpoints (Rust Backend)

| Method | Path                | Description |
|--------|---------------------|-------------|
| POST   | `/orchestrate`      | Run orchestration with a prompt |
| GET    | `/reports/:id`      | Retrieve a previous report |
| GET    | `/health`           | Health check |

**Example Response**

```json
{
  "success": true,
  "attempts_used": 2,
  "final_html": "<html>...</html>",
  "wcag_aa_score": 87.5,
  "wcag_aa_grade": "B",
  "validation_issues": []
}
```

## Integration Patterns

### 1. Recommended: tRPC + Rust REST

Use **tRPC** in Next.js for excellent type safety while calling a clean REST API from your Rust backend.

### 2. Simpler Alternative

Use plain `fetch` + TypeScript types if you prefer minimal setup.

### 3. High-Performance Option

Use **Connect** (gRPC-compatible) for maximum performance and strong contracts.

## Implementation Roadmap

### Phase 1: Foundation
- Expose `/orchestrate` endpoint in Rust using `AdvancedOrchestrator`
- Return `OrchestrationReport` as JSON
- Set up basic Next.js + tRPC project

### Phase 2: Core Integration
- Connect frontend to backend
- Build UI to trigger orchestration
- Display reports and WCAG AA scores

### Phase 3: Quality & Polish
- Add quality gate visualization
- Improve error handling
- Add authentication

## Authentication

- Use JWT tokens
- Validate tokens in the Rust backend on protected routes
- Pass tokens from Next.js on every request

## Best Practices

- Keep heavy orchestration logic in Rust (`web-forge`)
- Use `OrchestrationReport` as the primary data contract
- Leverage `should_fail_ci()` for quality gates
- Maintain clear separation between frontend and backend
- Use OpenTelemetry across both layers for observability

## Resources

- `docs/architecture.md` — System diagrams
- `docs/observability-reporting-ci.md` — Observability and CI patterns
