//! Schema Registry + Bridging Pass + Transfer Quality
//! v14.18.1 — Challenge provenance from Powrush high-road practice
//!
//! Implements deliberate principle extraction (high-road) alongside
//! similarity-triggered reuse (low-road), with mercy-gated provenance.
//! Parses `powrush_bridging_context_v1` / `powrush_bridging_batch_v1` from Powrush.
//! Optional challenge_* fields (v21.91.1+) enrich tags without breaking older payloads.
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortablePrincipleSchema {
    pub schema_id: String,
    pub principle: String,
    pub tags: Vec<String>,
    pub origin_session_id: Option<String>,
    pub origin_realm_id: Option<u8>,
    pub mercy_at_birth: f64,
    pub ethical_at_birth: f64,
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
    pub surface_label: String,
    pub challenge_id: Option<u64>,
    pub challenge_title: Option<String>,
    pub challenge_principle: Option<String>,
}

impl Default for BridgingContext {
    fn default() -> Self {
        Self {
            session_id: None,
            realm_id: None,
            decision_title: None,
            decision_type: None,
            mercy_factor: 0.0,
            ethical_score: 0.0,
            rbe_quality: 0.0,
            peaceful_rate: 0.0,
            abundance_velocity: 0.0,
            surface_label: String::new(),
            challenge_id: None,
            challenge_title: None,
            challenge_principle: None,
        }
    }
}

/// Wire envelope from Powrush `powrush_bridging_context_v1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushBridgingEnvelope {
    pub schema: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub realm_id: Option<u8>,
    #[serde(default)]
    pub decision_title: Option<String>,
    #[serde(default)]
    pub decision_type: Option<String>,
    pub mercy_factor: f64,
    pub ethical_score: f64,
    pub rbe_quality: f64,
    pub peaceful_rate: f64,
    pub abundance_velocity: f64,
    #[serde(default)]
    pub surface_label: String,
    #[serde(default)]
    pub decision_id: Option<u64>,
    #[serde(default)]
    pub tick: Option<u64>,
    #[serde(default)]
    pub challenge_id: Option<u64>,
    #[serde(default)]
    pub challenge_title: Option<String>,
    #[serde(default)]
    pub challenge_principle: Option<String>,
}

impl PowrushBridgingEnvelope {
    pub fn to_context(&self) -> BridgingContext {
        BridgingContext {
            session_id: self.session_id.clone(),
            realm_id: self.realm_id,
            decision_title: self.decision_title.clone(),
            decision_type: self.decision_type.clone(),
            mercy_factor: self.mercy_factor,
            ethical_score: self.ethical_score,
            rbe_quality: self.rbe_quality,
            peaceful_rate: self.peaceful_rate,
            abundance_velocity: self.abundance_velocity,
            surface_label: if self.surface_label.is_empty() {
                "powrush_bridging".into()
            } else {
                self.surface_label.clone()
            },
            challenge_id: self.challenge_id,
            challenge_title: self.challenge_title.clone(),
            challenge_principle: self.challenge_principle.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushBridgingBatch {
    pub schema: String,
    #[serde(default)]
    pub source: String,
    pub contexts: Vec<PowrushBridgingEnvelope>,
    #[serde(default)]
    pub exported_at_unix: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgingPassResult {
    pub extracted: Vec<PortablePrincipleSchema>,
    pub notes: Vec<String>,
    pub high_road_effort: bool,
}

pub fn bridging_pass(ctx: &BridgingContext) -> BridgingPassResult {
    let mut extracted = Vec::new();
    let mut notes = Vec::new();

    let challenge_tag = ctx.challenge_id.map(|id| format!("challenge_{}", id));

    if ctx.rbe_quality >= 0.65 && ctx.abundance_velocity >= 0.8 {
        let id = format!(
            "schema_rbe_alloc_{}",
            ctx.session_id.as_deref().unwrap_or("anon")
        );
        let principle = ctx
            .challenge_principle
            .clone()
            .filter(|p| p.to_lowercase().contains("allocation") || p.to_lowercase().contains("resource"))
            .unwrap_or_else(|| {
                "resource allocation under abundance pressure with sustainability bias".into()
            });
        let mut tags = vec!["rbe".into(), "abundance".into(), "allocation".into()];
        if let Some(ref t) = challenge_tag {
            tags.push(t.clone());
        }
        extracted.push(
            PortablePrincipleSchema::new(id, principle, tags, ctx.mercy_factor, ctx.ethical_score)
                .with_origin(ctx.session_id.clone(), ctx.realm_id),
        );
        notes.push("Extracted RBE allocation schema (high-road).".into());
    }

    if ctx.peaceful_rate >= 0.7 && ctx.ethical_score >= 0.65 {
        let id = format!(
            "schema_peaceful_resolve_{}",
            ctx.session_id.as_deref().unwrap_or("anon")
        );
        let principle = ctx
            .challenge_principle
            .clone()
            .filter(|p| p.to_lowercase().contains("peace") || p.to_lowercase().contains("resolution"))
            .unwrap_or_else(|| {
                "peaceful resolution under incomplete information with ethical priority".into()
            });
        let mut tags = vec!["peace".into(), "ethics".into(), "council".into()];
        if let Some(ref t) = challenge_tag {
            tags.push(t.clone());
        }
        extracted.push(
            PortablePrincipleSchema::new(id, principle, tags, ctx.mercy_factor, ctx.ethical_score)
                .with_origin(ctx.session_id.clone(), ctx.realm_id),
        );
        notes.push("Extracted peaceful-resolution schema (high-road).".into());
    }

    if ctx.mercy_factor >= 0.75 {
        let id = format!(
            "schema_mercy_priority_{}",
            ctx.session_id.as_deref().unwrap_or("anon")
        );
        let principle = ctx
            .challenge_principle
            .clone()
            .filter(|p| p.to_lowercase().contains("mercy"))
            .unwrap_or_else(|| "mercy-first prioritization when stakes are high".into());
        let mut tags = vec!["mercy".into(), "tolc".into(), "priority".into()];
        if let Some(ref t) = challenge_tag {
            tags.push(t.clone());
        }
        extracted.push(
            PortablePrincipleSchema::new(id, principle, tags, ctx.mercy_factor, ctx.ethical_score)
                .with_origin(ctx.session_id.clone(), ctx.realm_id),
        );
        notes.push("Extracted mercy-priority schema (high-road).".into());
    }

    // Explicit challenge principle when no threshold schema fired but practice context present
    if extracted.is_empty() {
        if let Some(ref principle) = ctx.challenge_principle {
            if ctx.mercy_factor >= 0.55 {
                let id = format!(
                    "schema_challenge_{}_{}",
                    ctx.challenge_id.unwrap_or(0),
                    ctx.session_id.as_deref().unwrap_or("anon")
                );
                let mut tags = vec!["challenge".into(), "high_road".into()];
                if let Some(ref t) = challenge_tag {
                    tags.push(t.clone());
                }
                extracted.push(
                    PortablePrincipleSchema::new(
                        id,
                        principle.clone(),
                        tags,
                        ctx.mercy_factor,
                        ctx.ethical_score,
                    )
                    .with_origin(ctx.session_id.clone(), ctx.realm_id),
                );
                notes.push("Extracted challenge-principle schema (high-road practice).".into());
            }
        }
    }

    if let Some(ref dtype) = ctx.decision_type {
        if dtype.to_lowercase().contains("harmony") || dtype.to_lowercase().contains("resource") {
            notes.push(format!(
                "Decision type '{}' mapped to portable RBE/harmony structure.",
                dtype
            ));
        }
    }

    if let Some(ref title) = ctx.challenge_title {
        notes.push(format!("Active practice challenge: {}", title));
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

/// Parse single `powrush_bridging_context_v1` JSON from Powrush-MMO.
pub fn parse_powrush_bridging_json(json: &str) -> Result<PowrushBridgingEnvelope, String> {
    let env: PowrushBridgingEnvelope = serde_json::from_str(json)
        .map_err(|e| format!("Mercy Gate (Truth): invalid bridging JSON: {}", e))?;
    if env.schema != "powrush_bridging_context_v1" {
        return Err(format!(
            "Mercy Gate (Truth): expected schema powrush_bridging_context_v1, got '{}'",
            env.schema
        ));
    }
    Ok(env)
}

/// Parse `powrush_bridging_batch_v1` JSON.
pub fn parse_powrush_bridging_batch_json(json: &str) -> Result<PowrushBridgingBatch, String> {
    let batch: PowrushBridgingBatch = serde_json::from_str(json)
        .map_err(|e| format!("Mercy Gate (Truth): invalid bridging batch JSON: {}", e))?;
    if batch.schema != "powrush_bridging_batch_v1" {
        return Err(format!(
            "Mercy Gate (Truth): expected schema powrush_bridging_batch_v1, got '{}'",
            batch.schema
        ));
    }
    if batch.contexts.is_empty() {
        return Err("Mercy Gate (Truth): bridging batch contains no contexts".into());
    }
    Ok(batch)
}

/// Parse → bridging_pass → ingest. Returns the pass result.
pub fn ingest_bridging_json(
    reg: &mut SchemaRegistry,
    json: &str,
) -> Result<BridgingPassResult, String> {
    let env = parse_powrush_bridging_json(json)?;
    let result = bridging_pass(&env.to_context());
    reg.ingest_bridging(result.clone());
    Ok(result)
}

/// Ingest every context in a bridging batch.
pub fn ingest_bridging_batch_json(
    reg: &mut SchemaRegistry,
    json: &str,
) -> Result<Vec<BridgingPassResult>, String> {
    let batch = parse_powrush_bridging_batch_json(json)?;
    let mut out = Vec::with_capacity(batch.contexts.len());
    for env in &batch.contexts {
        let result = bridging_pass(&env.to_context());
        reg.ingest_bridging(result.clone());
        out.push(result);
    }
    Ok(out)
}

// =============================================================================
// Schema registry
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
                    existing.reliability = (existing.reliability * 0.85 + 0.15).min(0.98);
                    existing.mercy_at_birth =
                        (existing.mercy_at_birth * 0.7 + schema.mercy_at_birth * 0.3).clamp(0.0, 1.0);
                })
                .or_insert(schema);
            n += 1;
        }
        n
    }

    pub fn retrieve_near(&self, tags: &[&str], min_reliability: f64) -> Vec<&PortablePrincipleSchema> {
        self.schemas
            .values()
            .filter(|s| s.reliability >= min_reliability)
            .filter(|s| tags.iter().any(|t| s.tags.iter().any(|st| st == t)))
            .collect()
    }

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
// Transfer quality metrics
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TransferQualityMetrics {
    pub near_transfer_success: u64,
    pub near_transfer_attempts: u64,
    pub far_transfer_success: u64,
    pub far_transfer_attempts: u64,
    pub abstraction_success_rate: f64,
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
// Metacognitive scaffolding
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaPhase {
    Planning,
    Monitoring,
    Evaluation,
}

pub fn metacognitive_prompt(phase: MetaPhase, support_level: f64) -> Option<&'static str> {
    if support_level <= 0.05 {
        return None;
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

pub fn scaffold_support_level(dominant_reliability: f64) -> f64 {
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

    const FIXTURE_BRIDGING: &str = include_str!("../fixtures/bridging_context_high_mercy.json");

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
            challenge_id: Some(1),
            challenge_title: Some("Caps Across Climates".into()),
            challenge_principle: Some("resource allocation under uncertainty".into()),
        };
        let result = bridging_pass(&ctx);
        assert!(result.high_road_effort);
        assert!(!result.extracted.is_empty());
        assert!(result.extracted.iter().any(|s| s.tags.iter().any(|t| t == "challenge_1")));
        assert!(result.notes.iter().any(|n| n.contains("Caps Across Climates")));
    }

    #[test]
    fn parse_and_ingest_bridging_fixture() {
        let mut reg = SchemaRegistry::new();
        let result = ingest_bridging_json(&mut reg, FIXTURE_BRIDGING).unwrap();
        assert!(!result.extracted.is_empty());
        assert!(!reg.is_empty());
        let near = reg.retrieve_near(&["rbe"], 0.3);
        assert!(!near.is_empty());
    }

    #[test]
    fn reject_wrong_bridging_schema() {
        let bad = r#"{"schema":"nope","mercy_factor":0.9,"ethical_score":0.9,"rbe_quality":0.9,"peaceful_rate":0.9,"abundance_velocity":1.0,"surface_label":"x"}"#;
        assert!(parse_powrush_bridging_json(bad).is_err());
    }

    #[test]
    fn mercy_gate_rejects_low_birth_schema() {
        let mut reg = SchemaRegistry::new();
        let weak = PortablePrincipleSchema::new("weak", "test principle", vec![], 0.2, 0.2);
        reg.schemas.insert(weak.schema_id.clone(), weak);
        assert!(reg.try_apply("weak", true, 0.5).is_err());
        assert_eq!(reg.total_mercy_rejects, 1);
    }

    #[test]
    fn scaffold_fades_with_reliability() {
        assert!(scaffold_support_level(0.9) < scaffold_support_level(0.4));
        assert!(metacognitive_prompt(MetaPhase::Planning, 0.0).is_none());
    }
}
