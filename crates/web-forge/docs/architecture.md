# web-forge Architecture

This document provides visual overviews of the `web-forge` system components and flows.

## 1. Overall System Architecture

```mermaid
graph TD
    A[Prompt] --> B[AdvancedOrchestrator]
    B --> C[PlanningStrategy]
    B --> D[ComponentAwareGenerator]
    B --> E[HtmlValidator]
    B --> F[OrchestrationReport]
    C --> G[DefaultPlanning / SemanticPlanning]
    E --> H[WCAG AA Scoring]
    F --> I[should_fail_ci]
    B --> J[Observability]
    J --> K[Tracing + Metrics]
```

## 2. Orchestration Flow

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Planner
    participant Generator
    participant Validator
    participant Reporter

    User->>Orchestrator: orchestrate(prompt)
    Orchestrator->>Planner: plan()
    Planner-->>Orchestrator: prioritized components
    Orchestrator->>Generator: generate()
    Orchestrator->>Validator: validate()
    loop Refinement
        Validator-->>Orchestrator: issues
        Orchestrator->>Generator: refine()
    end
    Orchestrator->>Reporter: build report
    Reporter-->>User: OrchestrationResult + Report
```

## 3. Observability + Reporting Pipeline

```mermaid
graph LR
    A[Orchestration Run] --> B[Tracing Spans]
    A --> C[Metrics Recording]
    A --> D[WCAG AA Score]
    B --> E[OpenTelemetry Collector]
    C --> E
    D --> F[OrchestrationReport]
    F --> G[should_fail_ci]
    G --> H[CI Pipeline]
```

## 4. CI Integration Flow

```mermaid
graph TD
    A[Push / PR] --> B[GitHub Actions]
    B --> C[Matrix: OS + Rust]
    C --> D[Build + Test]
    D --> E[Generate Report]
    E --> F[Quality Gate]
    F -->|Pass| G[Merge Allowed]
    F -->|Fail| H[Block PR]
    E --> I[Upload Artifact]
```
