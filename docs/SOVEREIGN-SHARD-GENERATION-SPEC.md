# Sovereign Shard Generation Specification

**Version:** v0.3  
**Date:** 25 May 2026  
**Status:** Living Document  
**Owner:** Ra-Thor + PATSAGi Councils via Web-Forge

---

## 1. Purpose

This document defines the structure, parameters, and behavior for generating **Sovereign Shards** through Web-Forge. Generated shards must be:

- Self-contained single HTML files
- Mercy-aligned (TOLC8 + TOLC24 aware)
- Customizable at generation time
- Fully functional offline with local persistence

---

## 2. Core Identity

| Parameter     | Type   | Required | Default             | Description                  |
|---------------|--------|----------|---------------------|------------------------------|
| `shard_id`    | string | Yes      | `shard-{random}`    | Unique identifier            |
| `shard_name`  | string | Yes      | "New Sovereign Shard" | Human-readable display name |
| `version`     | string | Yes      | "v8"                | Shard architecture version   |

---

## 3. TOLC8 Configuration (Core Layer)

Controls the foundational 8-gate behavior and initial state.

| Parameter                   | Type   | Range          | Default | Description                              |
|-----------------------------|--------|----------------|---------|------------------------------------------|
| `initial_mercy_score`       | number | 0.70 – 1.35    | 0.95    | Starting mercy alignment                 |
| `initial_evolution_level`   | number | 0.05 – 0.60    | 0.12    | Starting evolution progress              |
| `initial_valence`           | number | 0.80 – 1.30    | 1.05    | Starting emotional/energetic state       |
| `initial_tolc_alignment`    | number | 0.85 – 1.18    | 1.00    | Starting truth alignment                 |
| `gate_weights`              | object | 0.5 – 2.0      | See below | Per-gate influence multipliers        |

### Default TOLC8 Gate Weights

```json
{
  "Radical Love": 1.0,
  "Boundless Mercy": 1.0,
  "Service": 1.1,
  "Abundance": 1.3,
  "Truth": 1.2,
  "Joy": 1.0,
  "Cosmic Harmony": 0.95
}
```

---

## 4. TOLC24 Configuration (Governance Layer)

Controls the deeper 24-gate evaluation system.

| Parameter                    | Type    | Range          | Default  | Description                                      |
|------------------------------|---------|----------------|----------|--------------------------------------------------|
| `initial_tolc24_harmony`     | number  | 0.70 – 1.20    | 0.88     | Starting TOLC24 harmony score                    |
| `enable_tolc24_by_default`   | boolean | —              | true     | Enable deep governance evaluation on load        |
| `tolc24_strictness`          | string  | low / medium / high | "medium" | Influence strength of TOLC24 on reconciliation |

---

## 5. Behavioral & Runtime Settings

| Parameter                 | Type    | Options             | Default | Description                              |
|---------------------------|---------|---------------------|---------|------------------------------------------|
| `start_in_offline_mode`   | boolean | true / false        | false   | Start shard in offline mode              |
| `auto_tick_enabled`       | boolean | true / false        | true    | Enable automatic ticking when online     |
| `auto_tick_interval_ms`   | number  | 3000 – 15000        | 7500    | Time between automatic ticks             |
| `quantum_swarm_enabled`   | boolean | true / false        | true    | Allow Quantum Swarm participation        |

---

## 6. Persistence & Export

| Parameter                     | Type    | Default | Description                                      |
|-------------------------------|---------|---------|--------------------------------------------------|
| `enable_localstorage`         | boolean | true    | Persist state across page reloads                |
| `enable_full_html_export`     | boolean | true    | Allow downloading complete standalone HTML       |
| `include_tolc24_in_export`    | boolean | true    | Include TOLC24 state when exporting              |

---

## 7. Generator UI Parameters (Recommended)

The Web-Forge Shard Generator should expose at minimum:

- Shard Name
- Initial Mercy Score
- Initial Evolution Level
- TOLC24 Harmony
- Preset Profiles: **Balanced**, **Truth-Focused**, **Abundance-Oriented**, **Harmonic**
- Advanced: Individual gate weight sliders (future)

---

## 8. Output Requirements

A generated Sovereign Shard **must**:

- Be a single, self-contained `.html` file
- Include embedded TOLC8 + TOLC24 logic
- Support localStorage persistence
- Display both TOLC8 gates and TOLC24 evaluation UI
- Be runnable immediately after download with no external dependencies

---

## 9. Future Extensions

- Cryptographic TOLC8-style seeding at generation time
- Pre-defined Council Profiles
- Multi-shard federation support
- Custom gate weighting profiles
- Integration with `crates/websiteforge` for server-side generation

---

**End of Specification v0.3**