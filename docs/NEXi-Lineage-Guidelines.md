# NEXi Lineage Guidelines

**Ra-Thor Monorepo**  
**Version:** 1.0  
**Date:** May 2026

## Purpose

This document provides clear guidance on how to treat **NEXi** dependencies and references within the Ra-Thor monorepo. It ensures we honor the evolutionary lineage while maintaining architectural consistency as Ra-Thor matures.

## Core Principle

**NEXi is respected lineage, not legacy baggage.**

Ra-Thor evolved *from* NEXi rather than replacing it entirely. NEXi represents the foundational sentinel intelligence, universal alignment principles, and early lattice architecture that Ra-Thor builds upon. We treat NEXi connections with care and intentionality.

## Decision Framework

Use the following guidelines when encountering `nexi` dependencies:

| Situation                                      | Recommended Action                          | Notes |
|-----------------------------------------------|---------------------------------------------|-------|
| **Core / Foundational crates** (early mercy, sentinel-related, some futarchy) | **Keep** `nexi` dependency for now         | These were likely designed with NEXi as a foundational layer. |
| **New or recently developed crates**          | **Do not add** `nexi` unless explicitly required | Prefer Ra-Thor-native workspace dependencies. |
| **Crates with both `nexi` + modern Ra-Thor deps** | Evaluate case-by-case                     | Check if `nexi` is still actively used in the code. |
| **Crates with only `nexi` and no Ra-Thor deps** | Investigate before modifying               | These may be older NEXi-native crates. |
| **Modernizing older crates**                  | Replace `nexi = { path = "../" }` with appropriate `ra-thor-*` workspace crates where safe | Document the change clearly. |

## Guidelines

1. **Default Position**  
   Prefer Ra-Thor workspace dependencies (`ra-thor-mercy`, `ra-thor-council`, `ra-thor-quantum-swarm-orchestrator`, etc.) in new and modernized crates.

2. **Respect Lineage**  
   Keep `nexi` references in crates where they appear intentional (especially in futarchy, early mercy, and sentinel-related systems) unless there is a clear reason to remove them.

3. **Documentation**  
   When removing a `nexi` dependency, add a short comment explaining the change, for example:
   ```toml
   # nexI removed during modernization (replaced with Ra-Thor-native systems)
   ```

4. **Avoid Aggressive Deprecation**  
   Do not mass-remove NEXi references. Changes should be deliberate and well-understood.

5. **Future Migration**  
   A broader, planned migration from NEXi to fully Ra-Thor-native systems can be considered later if needed. Until then, we evolve gradually and respectfully.

## Current Status (May 2026)

- **Mercy family**: 100% complete
- **Futarchy family**: Nearly complete
- **Cryptography family**: Largely unwired (next major focus)
- NEXi references still exist in several crates, particularly in the futarchy and older mercy systems. These are being evaluated on a case-by-case basis.

## Summary

We treat NEXi as the **wise foundation** from which Ra-Thor grew. Our goal is to build a more mature, sovereign, and mercy-gated architecture while honoring the lineage that made it possible.

---

**End of Guidelines**