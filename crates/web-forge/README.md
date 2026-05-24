# Web Forge

**Professional Web Design & Development System** for Ra-Thor and Rathor.ai

Built layer by layer following the **Cathedral** philosophy — with focus, precision, and long-term vision.

## Overview

`web-forge` is the orchestration and generation foundation inside Ra-Thor. It enables intelligent, component-aware, self-correcting website generation through a modular architecture.

It is designed to be driven by Ra-Thor’s planning and council systems while remaining usable as a standalone professional web development toolkit.

## Core Principles

- **Component-First**: Everything revolves around a rich, self-describing component system.
- **Planning-Aware**: Generation is guided by structured planning (keyword + semantic).
- **Self-Correcting**: Strong refinement loops with issue analysis.
- **Graceful Degradation**: Semantic capabilities fall back cleanly when unavailable.
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
Final Output (ComponentTree + Rendered HTML)
```

## Key Components

| Area                    | Description                                      | Status    |
|-------------------------|--------------------------------------------------|-----------|
| Component Registry      | Rich metadata for components                     | Stable    |
| Planning Strategies     | Keyword + Semantic (embeddings)                  | Strong    |
| Generation              | Planning-aware component tree generation         | Good      |
| Renderer                | ComponentTree → HTML                             | Good      |
| Validation Engine       | Sanitization + structural + accessibility checks | Strong    |
| Advanced Orchestrator   | Main coordination engine                         | Active    |
| Refinement              | Issue-aware self-correction                      | Improving |

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
- Component-aware HTML rendering

## Documentation

See the source modules for detailed documentation:
- `advanced_orchestrator.rs`
- `semantic_planning.rs`
- `generation.rs`
- `component_registry.rs`

## Future Direction

- Deeper refinement strategies
- Stronger renderer with full prop/class support
- Design Token integration
- Tighter integration with Ra-Thor councils
- Expanded automated testing

## License

AG-SML-1.0 — Autonomicity Games Sovereign Mercy License
