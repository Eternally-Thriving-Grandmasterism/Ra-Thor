# Absolute Pure Truth Distillation

> This document captures the distilled learnings, principles, and architectural truths discovered during the development of web-forge.

## Core Philosophy

**Cathedral Approach**
- Build with care, precision, and long-term vision.
- Quality and maintainability over speed.
- Every layer should strengthen the ones above it.

## Key Learnings from Competitor Analysis

- Most AI site builders optimize for speed of creation, often at the cost of long-term quality and maintainability.
- Professional users need **clean, exportable code** and deep customization.
- Validation is a major gap in current AI generation tools.
- Strong design systems and component libraries are still underserved in AI builders.

## Architectural Truths

### 1. Validation is Non-Negotiable
AI generation without strong validation produces fragile output. Validation must be a first-class citizen in the architecture.

### 2. Components + Tokens as Foundation
A robust, token-driven component system enables both AI generation and manual professional control.

### 3. Hybrid Control is Powerful
The best experience combines:
- Natural language for speed
- Visual editing for intuition
- Code access for precision

### 4. Maintainability > Novelty
Generated websites must remain editable and maintainable over time. This is a core differentiator.

## Tooling Principles

- Strong local quality gates (ESLint, Husky, Commitlint)
- Unified build system
- CI validation
- Clear separation between generation and validation layers

## Commitment

As we build, we commit not only code, but the distilled truth behind the decisions. This document will evolve alongside the project.