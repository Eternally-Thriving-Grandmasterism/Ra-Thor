# Property-Based Testing Guide — web-forge

This document explains how and why we use **property-based testing** in `web-forge`.

## What is Property-Based Testing?

Instead of writing tests with fixed examples, we define **general properties** (invariants) that should hold for a wide range of inputs. The testing framework generates many random inputs (including edge cases) to try to falsify those properties.

If a property fails, the framework attempts to **shrink** the failing input to the smallest reproducible case.

## Why We Use It

- Finds bugs that example-based tests often miss
- Increases confidence in critical logic (orchestration, scoring, quality gates)
- Complements our existing test suite
- Aligns with our philosophy of **confidence over coverage**

## Tooling

We use **`proptest`** — the most mature and actively maintained property-based testing library for Rust.

```toml
[dev-dependencies]
proptest = "1.5"
```

## How We Apply It

We focus property-based testing on high-value areas:

- WCAG AA scoring invariants (score range, valid grades)
- Orchestration result structure
- Quality gate behavior (`should_fail_ci`)
- Refinement loop guarantees

### Example

```rust
proptest! {
    #[test]
    fn score_is_always_valid(html in any::<String>()) {
        let result = calculate_wcag_aa_score(&html);
        prop_assert!((0.0..=100.0).contains(&result.score));
    }
}
```

## Best Practices

- Start with simple properties on critical paths
- Use shrinking effectively
- Combine with example-based tests
- Keep properties focused and readable
- Document why each property matters

## When to Use

Use property-based testing when:
- There are clear invariants
- Input space is large
- Traditional tests feel insufficient
- We want higher confidence in core logic

## Summary

Property-based testing is a powerful tool that helps us build more robust and trustworthy systems. We use it thoughtfully, focused on areas that matter most to `web-forge`.
