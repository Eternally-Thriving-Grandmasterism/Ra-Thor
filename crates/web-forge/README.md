# Web Forge

**Professional Web Design & Development System for Ra-Thor**

> "Building websites brick by brick, like constructing a cathedral — with focus, love, precision, and long-term vision."

`web-forge` is a dedicated system inside the Ra-Thor monorepo for creating high-quality, maintainable, and mercy-aligned websites. It serves both Ra-Thor’s own presence and future projects under Rathor.ai.

## Philosophy

- **Cathedral Approach**: Every layer is built with care. Quality over speed. Structure before features.
- **Professional Standards**: Clean HTML, reliable interactivity, strong accessibility, and automated validation.
- **Mercy-Aligned Development**: Tools and processes that reduce friction and support truthful, focused work.
- **Reusable & Layered**: Design tokens → Components → Templates → Full sites.
- **Multi-Language Native**: Internationalization is a core concern from the beginning.

## Goals

- Prevent structural issues (e.g. malformed accordions, markdown leakage) through automated validation.
- Provide reliable multi-language switching.
- Offer a professional design system aligned with Ra-Thor’s aesthetic and values.
- Enable consistent, high-quality website development across projects.

## Current Status

This crate is in early foundation stage. The first deliverable is a clean, validated landing page template with built-in structural validation and language switching support.

## Structure

```
crates/web-forge/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── design_system/
│   ├── validation/
│   └── i18n/
└── examples/
    └── rathor_current_site.rs
```

## How to Contribute

All contributions should follow the AG-SML license and the broader mercy-aligned principles of the Ra-Thor project.

---

**Next layers will include:**
- Design tokens and component system
- Stronger automated HTML validation
- Rust-based template generation
- Full multi-language support tooling