# Property-Based Testing Guide — web-forge

This document explains how and why we use **property-based testing** in `web-forge`, with focus on `proptest` shrinking combinators.

## What is Property-Based Testing?

Property-based testing allows us to define general **properties** (invariants) instead of specific examples. `proptest` generates many inputs and tries to find counterexamples.

## Why We Use `proptest`

- Strong shrinking capabilities
- Rich set of combinators
- Good integration with Rust testing ecosystem
- Helps us build confidence in critical logic (orchestration, scoring, quality gates)

## Key Shrinking Combinators

### 1. `prop::collection::vec(strategy, size_range)`

Generates vectors and shrinks both length and elements.

```rust
prop::collection::vec(any::<String>(), 0..20)
```

### 2. `prop::string::string_regex(regex)`

Generates strings matching a regex. Shrinking stays within valid strings.

```rust
prop::string::string_regex(r#"<[^>]*>"#).unwrap()
```

### 3. `prop::option::of(strategy)`

Wraps a strategy in `Option`. Tries `None` first when shrinking.

```rust
prop::option::of(any::<String>())
```

### 4. `any::<T>()`

Default strategy with good built-in shrinking for most types.

### 5. `.prop_map(f)`

Transforms values while preserving shrinking behavior.

## Best Practices in web-forge

- Use bounded collections for better shrinking performance
- Prefer `string_regex` when generating structured input (e.g. HTML)
- Use `prop::option::of` for optional fields
- Combine with example-based tests for maximum confidence

## Current Usage

We apply property-based testing to:
- WCAG AA scoring invariants
- Orchestration result structure
- Quality gate behavior

See `src/validation/accessibility_scorer.rs` for examples.

## Summary

`proptest` shrinking combinators give us powerful tools to write robust, maintainable property-based tests. We use them thoughtfully to increase confidence in `web-forge`.
