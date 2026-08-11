//! Lattice Conductor soft hook — SchemaRegistry query surface
//! v14.18.0
//!
//! Zero hard dependency on lattice-conductor crates. Conductor / PATSAGi
//! adapters call these pure functions with a shared SchemaRegistry.
//!
//! AG-SML v1.0 | TOLC 8 | Contact: info@Rathor.ai

use crate::schema_registry::{
    PortablePrincipleSchema, SchemaRegistry, TransferQualityMetrics,
};
use serde::{Deserialize, Serialize};

/// Query request from Lattice Conductor / council deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorSchemaQuery {
    /// Prefer low-road tag retrieval when true; high-road principle search when false.
    pub near_road: bool,
    pub tags: Vec<String>,
    pub principle_query: Option<String>,
    pub min_reliability: f64,
    pub max_results: usize,
}

impl Default for ConductorSchemaQuery {
    fn default() -> Self {
        Self {
            near_road: true,
            tags: vec!["rbe".into(), "mercy".into()],
            principle_query: None,
            min_reliability: 0.4,
            max_results: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorSchemaHit {
    pub schema_id: String,
    pub principle: String,
    pub tags: Vec<String>,
    pub reliability: f64,
    pub mercy_at_birth: f64,
    pub origin_realm_id: Option<u8>,
}

impl From<&PortablePrincipleSchema> for ConductorSchemaHit {
    fn from(s: &PortablePrincipleSchema) -> Self {
        Self {
            schema_id: s.schema_id.clone(),
            principle: s.principle.clone(),
            tags: s.tags.clone(),
            reliability: s.reliability,
            mercy_at_birth: s.mercy_at_birth,
            origin_realm_id: s.origin_realm_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorSchemaQueryResult {
    pub hits: Vec<ConductorSchemaHit>,
    pub road: String,
    pub registry_size: usize,
}

/// Lattice Conductor entry: query portable principles without owning the registry.
pub fn conductor_query_schemas(
    reg: &SchemaRegistry,
    query: &ConductorSchemaQuery,
) -> ConductorSchemaQueryResult {
    let limit = query.max_results.max(1).min(64);
    let hits: Vec<ConductorSchemaHit> = if query.near_road {
        let tag_refs: Vec<&str> = query.tags.iter().map(|s| s.as_str()).collect();
        reg.retrieve_near(&tag_refs, query.min_reliability)
            .into_iter()
            .take(limit)
            .map(ConductorSchemaHit::from)
            .collect()
    } else {
        let q = query
            .principle_query
            .as_deref()
            .unwrap_or("allocation");
        reg.retrieve_far(q, query.min_reliability)
            .into_iter()
            .take(limit)
            .map(ConductorSchemaHit::from)
            .collect()
    };

    ConductorSchemaQueryResult {
        hits,
        road: if query.near_road {
            "near".into()
        } else {
            "far".into()
        },
        registry_size: reg.len(),
    }
}

/// Conductor apply: mercy-gated schema activation with far/near flag.
pub fn conductor_try_apply_schema(
    reg: &mut SchemaRegistry,
    schema_id: &str,
    is_far: bool,
    mercy_floor: f64,
) -> Result<ConductorSchemaHit, String> {
    let applied = reg.try_apply(schema_id, is_far, mercy_floor)?;
    Ok(ConductorSchemaHit::from(applied))
}

/// Snapshot transfer quality for Conductor dashboards / PATSAGi notes.
pub fn conductor_transfer_quality_snapshot(
    reg: &SchemaRegistry,
    abstraction_rate: f64,
    meta_compliance: f64,
) -> TransferQualityMetrics {
    TransferQualityMetrics::from_registry(reg, abstraction_rate, meta_compliance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_registry::{bridging_pass, BridgingContext};

    #[test]
    fn conductor_near_query_hits() {
        let mut reg = SchemaRegistry::new();
        let result = bridging_pass(&BridgingContext {
            session_id: Some("c1".into()),
            realm_id: Some(2),
            decision_title: None,
            decision_type: Some("ResourcePolicy".into()),
            mercy_factor: 0.9,
            ethical_score: 0.9,
            rbe_quality: 0.9,
            peaceful_rate: 0.9,
            abundance_velocity: 1.3,
            surface_label: "t".into(),
        });
        reg.ingest_bridging(result);

        let q = ConductorSchemaQuery {
            near_road: true,
            tags: vec!["rbe".into()],
            principle_query: None,
            min_reliability: 0.3,
            max_results: 4,
        };
        let out = conductor_query_schemas(&reg, &q);
        assert_eq!(out.road, "near");
        assert!(!out.hits.is_empty());
    }
}
