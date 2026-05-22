//! Connection between Sovereign Reflexion Core and existing mercy_gating module

/// # Integration Strategy
///
/// The Sovereign Reflexion Core (SRC) is designed to work *with*,
/// not replace, the existing `mercy_gating` module.
///
/// ## Responsibilities
///
/// - `mercy_gating` remains the core evaluation engine (MercyVerdict, MaatKpi, etc.)
/// - SRC acts as the **orchestration + reflexion + historical + escalation layer**
///
/// ## Specific Connections
///
/// 1. SRC reuses `mercy_gating::MercyVerdict` and related types.
/// 2. SRC calls `mercy_gating::self_referential_mercy_evaluation()` internally.
/// 3. `MercyEvaluationHistory` is owned by SRC but exposed to SovereignHealthMonitor.
/// 4. When SRC decides to escalate, it provides rich context to future PATSAGi systems.

pub use crate::mercy_gating::{MercyVerdict, MercyGateLevel, self_referential_mercy_evaluation};