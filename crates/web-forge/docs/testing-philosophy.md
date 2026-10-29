# Testing Philosophy — web-forge

This document outlines the testing principles and approach used in `web-forge`.

## Core Principles

### 1. Confidence Over Coverage
We prioritize tests that give us real confidence in behavior, especially around:
- Orchestration flows
- Quality gates and reporting
- Accessibility (WCAG AA) scoring and validation
- Observability instrumentation

We prefer meaningful tests over high coverage numbers.

### 2. Readability and Intent
Tests should be easy to read and understand. We use clear naming and section comments so that the *intent* of each test is obvious.

Example:
```rust
// ------------------------------------------------------------------------- 
// Reporting & Quality Gates
// -------------------------------------------------------------------------
```

### 3. Resistance to Brittle Changes
We design tests to verify behavior and contracts rather than internal implementation details. This makes tests more resilient when we refactor.

### 4. Support for CI Quality Gates
Many of our tests exist to support reliable CI decisions through `OrchestrationReport` and `should_fail_ci()`. Tests in this area are especially important.

### 5. Observability and Accessibility Matter
We actively test:
- WCAG AA scoring correctness
- Accessibility validation rules
- That observability instrumentation (tracing/metrics) does not break core flows

## Test Organization

- Tests live close to the code they verify (usually in `#[cfg(test)] mod tests`).
- We group related tests with clear section comments.
- We avoid excessive fragmentation of test modules.

## What We Test

| Area                    | Priority | Notes |
|-------------------------|----------|-------|
| Orchestration behavior  | High     | Core flows and edge cases |
| Refinement logic        | High     | Max attempts, issue handling |
| WCAG AA Scoring         | High     | Correct scoring and grading |
| Quality Gates           | High     | `should_fail_ci` and reporting |
| Observability           | Medium   | Ensure spans and metrics paths work |
| Semantic Planning       | Medium   | Path availability and fallback |

## What We Avoid

- Overly brittle tests that break on minor refactors
- Testing implementation details instead of behavior
- Low-value tests that don’t increase confidence

## Philosophy Summary

> We test to build **confidence**, not just coverage. 
> Tests should be **readable**, **intentional**, and **support the quality gates** that protect the system in CI.

This philosophy helps us maintain a high-quality, sustainable codebase as `web-forge` continues to evolve.
