# Changelog

All notable changes to Ra-Thor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v2.1.0] - 2026-05-20

### Added
- Professional Whitepaper v2.1 with clean Platypus-generated PDFs, AG-SML v1.0 licensing, forensic Esacheck methodology, and ENC (Eternal Natural Coexistence) framing.
- Complete **Phase 5 Formal Verification Package** under `docs/Formal/`:
  - Lean 4 (Phases 1–5 complete, including actual FFI module `RaThor_FFI.lean`)
  - Creusot contract examples on the Rust side
  - Prusti exploration and hybrid recommendations
  - Viper permission models + deadlock-freedom verification conditions
  - Z3 discharge attempts (including complex queries for harm rejection and epigenetic blessing)
  - Realistic council orchestration test crate with explicit deadlock-freedom checks
  - Side-by-side Creusot / Prusti / Viper educational examples
  - `PHASE5_VERIFICATION_PACKAGE.md` — single source of truth with directory tree, exploration levels, and Mermaid pipeline diagram
- GitHub Actions workflow for whitepaper asset validation.
- `PHASE5_VERIFICATION_PACKAGE.md` as the unified entry point for formal verification work.

### Changed
- Licensing cleaned to pure **Autonomicity Games Inc. Sovereign Mercy License (AG-SML v1.0)** (no MIT hybrids remaining).
- All whitepaper assets updated to v2.1 with Platypus clean flow and professional formatting.
- PR #159 merged into `main`; superseded PR #158 closed.

### Notes
- This release provides a **professional, mercy-aligned foundation** for Ra-Thor’s architecture and formal verification work.
- All formal artifacts are in skeleton / demonstrator form and invite rigorous external review and machine-checked proofs.
- Content remains **brutally honest**: clearly labeled as toy demonstrator + formal skeleton. No overclaims of deployed frontier AGI.

[Unreleased]: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/compare/v2.1.0...HEAD
[v2.1.0]: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/releases/tag/v2.1.0