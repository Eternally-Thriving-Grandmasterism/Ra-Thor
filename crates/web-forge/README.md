# Web Forge

**Professional Web Design & Development System** for Ra-Thor and Rathor.ai

Built layer by layer following the **Cathedral** philosophy — with focus, precision, and long-term vision.

## Overview

`web-forge` is the orchestration and generation foundation inside Ra-Thor. It enables intelligent, component-aware, self-correcting website generation with strong observability, accessibility, and CI integration.

It is designed to be driven by Ra-Thor’s planning and council systems while remaining usable as a standalone professional web development toolkit.

## Core Principles

- **Component-First**: Everything revolves around a rich, self-describing component system.
- **Planning-Aware**: Generation is guided by structured planning (keyword + semantic).
- **Self-Correcting**: Strong refinement loops with issue analysis.
- **Graceful Degradation**: Semantic capabilities fall back cleanly when unavailable.
- **Quality & Observability**: Built-in tracing, metrics, WCAG AA scoring, and automated reporting.
- **Extensible by Design**: Built for future intelligence, new strategies, and deep Ra-Thor integration.

## Architecture

```
Prompt
   │
   ▼
PlanningStrategy (Default / Semantic)
   │
   ▼
PlanningResult (scored + prioritized components)
   │
   ▼
ComponentAwareGenerator
   │
   ▼
Validation + Refinement Loop
   │
   ▼
Final Output (ComponentTree + Rendered HTML + Report)
```

## Key Components

| Area                    | Description                                           | Status    |
|-------------------------|-------------------------------------------------------|-----------|
| Component Registry      | Rich metadata for components                          | Stable    |
| Planning Strategies     | Keyword + Semantic (embeddings)                       | Strong    |
| Generation              | Planning-aware component tree generation              | Good      |
| Renderer                | ComponentTree → HTML                                  | Good      |
| Validation Engine       | Sanitization + structural + WCAG AA accessibility     | Strong    |
| Advanced Orchestrator   | Main coordination engine                              | Active    |
| Observability           | Tracing + Metrics (OpenTelemetry)                     | Strong    |
| Reporting & CI          | Automated reports + quality gates                     | Strong    |

## Usage Example

```rust
use web_forge::orchestration::AdvancedOrchestrator;

let orchestrator = AdvancedOrchestrator::new()
    .with_max_attempts(3)
    .with_semantic_planning("sk-...".to_string());

let result = orchestrator.orchestrate("Create a beautiful primary call-to-action");

if result.success {
    println!("Generated HTML:\n{}", result.final_html.unwrap_or_default());
}
```

## Current Capabilities

- Multi-intent detection during planning
- Relevance scoring and component prioritization
- Planning-aware generation
- Semantic embeddings with automatic fallback
- Structured refinement with issue categorization
- WCAG AA accessibility scoring and validation
- Component-aware HTML rendering
- Built-in observability (tracing + metrics)
- Automated reporting with `should_fail_ci` for CI gates

## Documentation

See the `docs/` folder for detailed guides:
- `docs/architecture.md` — System diagrams and flows
- `docs/observability-reporting-ci.md` — Observability, reporting, and CI integration

## Future Direction

- Deeper refinement strategies
- Stronger renderer with full prop/class support
- Design Token integration
- Tighter integration with Ra-Thor councils
- Expanded automated testing

## License

AG-SML-1.0 — Autonomicity Games Sovereign Mercy License
