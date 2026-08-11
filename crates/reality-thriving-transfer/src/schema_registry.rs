//! Schema Registry + Bridging Pass + Transfer Quality
//! v14.16.0 — High-road skill transfer for Ra-Thor / Powrush coupling
//!
//! Implements deliberate principle extraction (high-road) alongside
//! similarity-triggered reuse (low-road), with mercy-gated provenance.
//!
//! Theoretical anchors: Perkins & Salomon (low-road / high-road),
//! identical elements, schema theory, metacognitive scaffolding.
//!
//! AG-SML v1.0 | TOLC 8 Living Mercy Gates | Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Portable principle schemas
// =============================================================================

/// Deep-structure principle extracted from a concrete decision or session.
/// Surface features are discarded; only portable structure remains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortablePrincipleSchema {
    pub schema_id: String,
    /// Human/agent-readable principle (e.g. "resource allocation under uncertainty").
    pub principle: String,
    /// Optional tags for retrieval ("rbe", "harmony", "council", "combat").
    pub tags: Vec<String>,
    /// Origin session or decision id (provenance).
    pub origin_session_id: Option<String>,
    pub origin_realm_id: Option<u8>,
    /// Mercy / ethical score at extraction time — revalidated on application.
    pub mercy_at_birth: f64,
    pub ethical_at_birth: f64,
    /// Reliability grows with successful far/near reuses; fades on failure.
    pub reliability: f64,
    pub near_reuse_count: u64,
    pub far_reuse_count: u64,
    pub failed_reuse_count: u64,
    pub created_unix: u64,
    pub last_applied_unix: Option<u64>,
}

impl PortablePrincipleSchema {
    pub fn new(
        schema_id: impl Into<String>,
        principle: impl Into<String>,
        tags: Vec<String>,
        mercy: f64,
        ethical: f64,
    ) -> Self {
        Self {
            schema_id: schema_id.into(),
            principle: principle.into(),
            tags,
            origin_session_id: None,
            origin_realm_id: None,
            mercy_at_birth: mercy.clamp(0.0, 1.0),
            ethical_at_birth: ethical.clamp(0.0, 1.0),
            reliability: 0.55,
            near_reuse_count: 0,
            far_reuse_count: 0,
            failed_reuse_count: 0,
            created_unix: now_secs(),
            last_applied_unix: None,
        }
    }

    pub fn with_origin(mut self, session_id: Option<String>, realm_id: Option<u8>) -> Self {
        self.origin_session_id = session_id;
        self.origin_realm_id = realm_id;
        self
    }

    /// Mercy gate: refuse application if birth mercy is below floor.
    pub fn passes_mercy_floor(&self, floor: f64) -> bool {
        self.mercy_at_birth >= floor && self.ethical_at_birth >= floor * 0.9
    }
}

// =============================================================================
// Bridging pass (high-road extraction)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgingContext {
    pub session_id: Option<String>,
    pub realm_id: Option<u8>,
    pub decision_title: Option<String>,
    pub decision_type: Option<String>,
    pub mercy_factor: f64,
    pub ethical_score: f64,
    pub rbe_quality: f64,
    pub peaceful_rate: f64,
    pub abundance_velocity: f64,
    /// Surface label of the concrete situation (for contrast, not stored as principle).
    pub surface_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgingPassResult {
    pub extracted: Vec<PortablePrincipleSchema>,
    pub notes: Vec<String>,
    pub high_road_effort: bool,
}

/// Deliberate high-road extraction from a concrete council / session context.
/// Does not rely on surface similarity — forces principle abstraction.
pub fn bridging_pass(ctx: &BridgingContext) -> BridgingPassResult {
    let mut extracted = Vec::new();
    let mut notes = Vec::new();

    // Core abstractions from telemetry structure (not surface build orders).
    if ctx.rbe_quality >= 0.65 && ctx.abundance_velocity >= 0.8 {
        let id = format!(
            "schema_rbe_alloc_{}",
            ctx.session_id.as_deref().unwrap_or("anon")
        );
        extracted.push(
            PortablePrincipleSchema::new(
                id,
                "resource allocation under abundance pressure with sustainability bias",
                vec!["rbe".into(), "abundance".into(), "allocation".into()],
                ctx.mercy_factor,
                ctx.ethical_score,
            )
            .with_origin(ctx.session_id.clone(), ctx.realm_id),
        );
        notes.push("Extracted RBE allocation schema (high-road).".into());
    }

    if ctx.peaceful_rate >= 0.7 && ctx.ethical_score >= 0.65 {
        let id = format!(
            "schema_peaceful_resolve_{}",
            ctx.session_id.as_deref().unwrap_or("anon")
        );
        extracted.push(
            PortablePrincipleSchema::new(
                id,
                "peaceful resolution under incomplete information with ethical priority",
                vec!["peace".into(), "ethics".into(), "council".into()],
                ctx.mercy_factor,
                ctx.ethical_score,
            )
            .with_origin(ctx.session_id.clone(), ctx.realm_id),
        );
        notes.push("Extracted peaceful-resolution schema (high-road).".into());
    }

    if ctx.mercy_factor >= 0.75 {
        let id = format!(
            "schema_mercy_priority_{}",
            ctx.session_id.as_deref().unwrap_or("anon")
        );
        extracted.push(
            PortablePrincipleSchema::new(
                id,
                "mercy-first prioritization when stakes are high",
                vec!["mercy".into(), "tolc".into(), "priority".into()],
                ctx.mercy_factor,
                ctx.ethical_score,
            )
            .with_origin(ctx.session_id.clone(), ctx.realm_id),
        );
        notes.push("Extracted mercy-priority schema (high-road).".into());
    }

    if let Some(ref dtype) = ctx.decision_type {
        if dtype.to_lowercase().contains("harmony") || dtype.to_lowercase().contains("resource") {
            notes.push(format!(
                "Decision type '{}' mapped to portable RBE/harmony structure.",
                dtype
            ));
        }
    }

    if extracted.is_empty() {
        notes.push(
            "No strong principle extracted — surface conditions below abstraction thresholds."
                .into(),
        );
    }

    BridgingPassResult {
        high_road_effort: true,
        extracted,
        notes,
    }
}

// =============================================================================
// Schema registry (storage + low-road / high-road retrieval)
// =============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaRegistry {
    pub schemas: HashMap<String, PortablePrincipleSchema>,
    pub total_bridging_passes: u64,
    pub total_near_hits: u64,
    pub total_far_hits: u64,
    pub total_mercy_rejects: u64,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest_bridging(&mut self, result: BridgingPassResult) -> usize {
        self.total_bridging_passes = self.total_bridging_passes.saturating_add(1);
        let mut n = 0;
        for schema in result.extracted {
            self.schemas
                .entry(schema.schema_id.clone())
                .and_modify(|existing| {
                    // Strengthen reliability on re-extraction of same id lineage
                    existing.reliability = (existing.reliability * 0.85 + 0.15).min(0.98);
                    existing.mercy_at_birth =
                        (existing.mercy_at_birth * 0.7 + schema.mercy_at_birth * 0.3).clamp(0.0, 1.0);
                })
                .or_insert(schema);
            n += 1;
        }
        n
    }

    /// Low-road: tag overlap retrieval (similarity-triggered).
    pub fn retrieve_near(&self, tags: &[&str], min_reliability: f64) -> Vec<&PortablePrincipleSchema> {
        self.schemas
            .values()
            .filter(|s| s.reliability >= min_reliability)
            .filter(|s| tags.iter().any(|t| s.tags.iter().any(|st| st == t)))
            .collect()
    }

    /// High-road: principle-text / deep-structure search (deliberate mapping).
    pub fn retrieve_far(
        &self,
        principle_query: &str,
        min_reliability: f64,
    ) -> Vec<&PortablePrincipleSchema> {
        let q = principle_query.to_lowercase();
        self.schemas
            .values()
            .filter(|s| s.reliability >= min_reliability)
            .filter(|s| {
                s.principle.to_lowercase().contains(&q)
                    || s.tags.iter().any(|t| q.contains(&t.to_lowercase()))
            })
            .collect()
    }

    /// Apply a schema in a new context — revalidates mercy floor.
    pub fn try_apply(
        &mut self,
        schema_id: &str,
        is_far: bool,
        mercy_floor: f64,
    ) -> Result<&PortablePrincipleSchema, String> {
        let schema = self
            .schemas
            .get_mut(schema_id)
            .ok_or_else(|| format!("Schema '{}' not found", schema_id))?;

        if !schema.passes_mercy_floor(mercy_floor) {
            self.total_mercy_rejects = self.total_mercy_rejects.saturating_add(1);
            schema.failed_reuse_count = schema.failed_reuse_count.saturating_add(1);
            schema.reliability = (schema.reliability * 0.92).max(0.1);
            return Err(format!(
                "Mercy Gate: schema '{}' fails floor {:.2} (mercy={:.2})",
                schema_id, mercy_floor, schema.mercy_at_birth
            ));
        }

        schema.last_applied_unix = Some(now_secs());
        schema.reliability = (schema.reliability * 0.9 + 0.1).min(0.99);
        if is_far {
            schema.far_reuse_count = schema.far_reuse_count.saturating_add(1);
            self.total_far_hits = self.total_far_hits.saturating_add(1);
        } else {
            schema.near_reuse_count = schema.near_reuse_count.saturating_add(1);
            self.total_near_hits = self.total_near_hits.saturating_add(1);
        }
        Ok(self.schemas.get(schema_id).unwrap())
    }

    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }
}

// =============================================================================
// Transfer quality metrics (instrumentation)
// =============================================================================

/// Optional companion metrics for RTT / council telemetry.
/// Does not break powrush_telemetry_v1; travels beside it.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TransferQualityMetrics {
    /// Successful same-context / same-realm reuses.
    pub near_transfer_success: u64,
    pub near_transfer_attempts: u64,
    /// Successful cross-context / cross-realm reuses.
    pub far_transfer_success: u64,
    pub far_transfer_attempts: u64,
    /// How often bridging_pass produced ≥1 schema.
    pub abstraction_success_rate: f64,
    /// Fraction of deliberations that ran planning/monitoring/evaluation prompts.
    pub metacognitive_compliance: f64,
    pub bridging_passes_run: u64,
    pub schemas_in_registry: u64,
}

impl TransferQualityMetrics {
    pub fn near_rate(&self) -> f64 {
        if self.near_transfer_attempts == 0 {
            0.0
        } else {
            self.near_transfer_success as f64 / self.near_transfer_attempts as f64
        }
    }

    pub fn far_rate(&self) -> f64 {
        if self.far_transfer_attempts == 0 {
            0.0
        } else {
            self.far_transfer_success as f64 / self.far_transfer_attempts as f64
        }
    }

    pub fn from_registry(reg: &SchemaRegistry, abstraction_rate: f64, meta_compliance: f64) -> Self {
        Self {
            near_transfer_success: reg.total_near_hits,
            near_transfer_attempts: reg.total_near_hits + reg.total_mercy_rejects / 2,
            far_transfer_success: reg.total_far_hits,
            far_transfer_attempts: reg.total_far_hits + reg.total_mercy_rejects / 2,
            abstraction_success_rate: abstraction_rate.clamp(0.0, 1.0),
            metacognitive_compliance: meta_compliance.clamp(0.0, 1.0),
            bridging_passes_run: reg.total_bridging_passes,
            schemas_in_registry: reg.len() as u64,
        }
    }
}

// =============================================================================
// Metacognitive scaffolding prompts (fadable)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaPhase {
    Planning,
    Monitoring,
    Evaluation,
}

/// Soft prompt strings for council / agent loops. Support level 1.0 = full scaffold, 0.0 = faded.
pub fn metacognitive_prompt(phase: MetaPhase, support_level: f64) -> Option<&'static str> {
    if support_level <= 0.05 {
        return None; // fully faded — independent self-regulation expected
    }
    match phase {
        MetaPhase::Planning => Some(
            "What is the goal? Which portable principles might apply? What mercy constraints are active?",
        ),
        MetaPhase::Monitoring => Some(
            "Are we still aligned with mercy bounds? Is surface similarity misleading the mapping?",
        ),
        MetaPhase::Evaluation => Some(
            "Which principles transferred successfully? Which failed? What should update in the schema registry?",
        ),
    }
}

/// Fade support as reliability of dominant schema rises.
pub fn scaffold_support_level(dominant_reliability: f64) -> f64 {
    // High reliability → less external scaffolding
    (1.0 - dominant_reliability).clamp(0.0, 1.0)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridging_extracts_on_strong_signals() {
        let ctx = BridgingContext {
            session_id: Some("s1".into()),
            realm_id: Some(2),
            decision_title: Some("Verdant Cap".into()),
            decision_type: Some("ResourcePolicy".into()),
            mercy_factor: 0.88,
            ethical_score: 0.85,
            rbe_quality: 0.9,
            peaceful_rate: 0.92,
            abundance_velocity: 1.4,
            surface_label: "realm_2_council".into(),
        };
        let result = bridging_pass(&ctx);
        assert!(result.high_road_effort);
        assert!(!result.extracted.is_empty());
        assert!(result.extracted.iter().all(|s| s.passes_mercy_floor(0.5)));
    }

    #[test]
    fn registry_near_and_far_retrieval() {
        let mut reg = SchemaRegistry::new();
        let result = bridging_pass(&BridgingContext {
            session_id: Some("s2".into()),
            realm_id: None,
            decision_title: None,
            decision_type: None,
            mercy_factor: 0.9,
            ethical_score: 0.9,
            rbe_quality: 0.85,
            peaceful_rate: 0.8,
            abundance_velocity: 1.2,
            surface_label: "test".into(),
        });
        reg.ingest_bridging(result);
        assert!(!reg.is_empty());

        let near = reg.retrieve_near(&["rbe"], 0.3);
        assert!(!near.is_empty());

        let far = reg.retrieve_far("allocation", 0.3);
        assert!(!far.is_empty());
    }

    #[test]
    fn mercy_gate_rejects_low_birth_schema() {
        let mut reg = SchemaRegistry::new();
        let weak = PortablePrincipleSchema::new("weak", "test principle", vec![], 0.2, 0.2);
        reg.schemas.insert(weak.schema_id.clone(), weak);
        let err = reg.try_apply("weak", true, 0.5);
        assert!(err.is_err());
        assert_eq!(reg.total_mercy_rejects, 1);
    }

    #[test]
    fn scaffold_fades_with_reliability() {
        assert!(scaffold_support_level(0.9) < scaffold_support_level(0.4));
        assert!(metacognitive_prompt(MetaPhase::Planning, 0.8).is_some());
        assert!(metacognitive_prompt(MetaPhase::Planning, 0.0).is_none());
    }

    #[test]
    fn transfer_quality_from_registry() {
        let reg = SchemaRegistry::new();
        let m = TransferQualityMetrics::from_registry(&reg, 0.7, 0.85);
        assert_eq!(m.schemas_in_registry, 0);
        assert!((m.metacognitive_compliance - 0.85).abs() < 1e-9);
    }
}
