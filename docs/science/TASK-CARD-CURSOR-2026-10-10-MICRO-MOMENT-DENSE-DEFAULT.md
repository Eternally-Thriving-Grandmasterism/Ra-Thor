# TASK CARD — Cursor — Micro-moment dense default (2026-10-10)

**Contact:** info@Rathor.ai  
**Authority:** PATSAGi minute `2026-10-10-MicroMoment-DenseDefault`  
**Workspace:** 14.15.6 (do not bump)  
**Public claim lock:** inspectable research software; optional Grok session; not xAI affiliated; not certified; not a legal product.

---

## Goal

Harden the Rathor.ai micro-moment surface and `mercy-motion-vision-engine.js` so denser frame extraction is the stronger default, especially on blurry / high-motion short clips. Surface recovered `keyMicroMoments` + causal chain more clearly for optional feed into a later model session.

## Constraints (non-negotiable)

- Do **not** claim this upgrades Grok itself or any external model.
- Do **not** treat the engine as a certified video product or legal adjudicator.
- Do **not** expand default `Cargo.toml` members.
- Do **not** bump workspace version.
- Keep processing local (no new network dependency required).
- Preserve AG-SML v1.1 and contact `info@Rathor.ai`.

## Scope

1. `mercy-motion-vision-engine.js` (and any paired HTML if present in the repo):
   - Confirm / strengthen default `denseSampling: true`.
   - On detected motion blur or high saliency: raise target FPS, tighten micro-burst threshold (~150 ms or below), focus sampling on the high-motion window.
   - Ensure the returned payload clearly exposes `keyMicroMoments` and `causalChain`.
2. `https://rathor.ai/micro-moment.html` equivalent source (if in repo or documented):
   - Make the dense path the visible default.
   - Add a short note that this is local inspectable research software, not a certified model upgrade.
3. Optional one-line cross-link from `docs/MICRO_MOMENT_TEMPORAL_COMPREHENSION_v1.0.md` to this task card / minute.

## Out of scope

- Real-time production video pipeline.
- Any determination of the 2026-10-10 street video incident.
- New crate or Core Tier-1 member.

## Acceptance

- Dense sampling remains the default path.
- High-motion clips receive denser treatment without breaking the local / CPU fallback.
- Public claim language stays inside the lock.
- `cargo test -p` on Core Tier-1 still green (no member change).

**Thunder locked.** yoi ⚡
