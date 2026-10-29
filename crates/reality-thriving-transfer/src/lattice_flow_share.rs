//! Lattice Flow Share ingest — Powrush soft allocate envelopes
//! v14.19.0
//!
//! Parses `powrush_lattice_flow_share_v1` (Powrush-MMO v21.93.3+)
//! and maps into BridgingContext → SchemaRegistry high-road pass.
//!
//! No scarcity. Abundance direction only.
//! Contact: info@Rathor.ai | TOLC 8 | Yoi ⚡

use serde::{Deserialize, Serialize};

use crate::schema_registry::{
    bridging_pass, BridgingContext, BridgingPassResult, SchemaRegistry,
};

/// Wire envelope from Powrush `data/powrush_lattice_flow_share.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowrushLatticeFlowShare {
    pub schema: String,
    pub flow_total: f32,
    pub reserve_total: f32,
    pub choices_made: u32,
    #[serde(default)]
    pub last_path: Option<String>,
    #[serde(default)]
    pub mercy_note: String,
    #[serde(default)]
    pub exported_at_secs: f64,
}

impl PowrushLatticeFlowShare {
    pub fn to_bridging_context(&self) -> BridgingContext {
        let total = (self.flow_total + self.reserve_total).max(0.001);
        let flow_ratio = (self.flow_total / total) as f64;
        // Soft signals — always non-negative abundance framing
        let abundance_velocity = (0.6 + (total as f64).min(20.0) / 20.0 * 0.8).min(1.6);
        let rbe_quality = (0.55 + flow_ratio * 0.35 + (self.choices_made as f64).min(12.0) / 12.0 * 0.1)
            .clamp(0.0, 1.0);
        let mercy = 0.82_f64; // voluntary allocate is mercy-aligned by design
        let principle = match self.last_path.as_deref() {
            Some(p) if p.to_lowercase().contains("flow") => {
                "share surplus into the living lattice under abundance".to_string()
            }
            Some(p) if p.to_lowercase().contains("steward") || p.to_lowercase().contains("reserve") => {
                "steward surplus for future thriving without scarcity framing".to_string()
            }
            _ => "resource allocation under abundance with voluntary direction".to_string(),
        };

        BridgingContext {
            session_id: Some(format!("lattice_flow_{}", self.choices_made)),
            realm_id: None,
            decision_title: self.last_path.clone(),
            decision_type: Some("LatticeFlowShare".into()),
            mercy_factor: mercy,
            ethical_score: mercy,
            rbe_quality,
            peaceful_rate: 0.9,
            abundance_velocity,
            surface_label: "powrush_lattice_flow_share".into(),
            challenge_id: Some(1),
            challenge_title: Some("Caps Across Climates".into()),
            challenge_principle: Some(principle),
        }
    }
}

pub fn parse_powrush_lattice_flow_share_json(
    json: &str,
) -> Result<PowrushLatticeFlowShare, String> {
    let env: PowrushLatticeFlowShare = serde_json::from_str(json)
        .map_err(|e| format!("Mercy Gate (Truth): invalid lattice flow share JSON: {}", e))?;
    if !env.schema.starts_with("powrush_lattice_flow_share") {
        return Err(format!(
            "Mercy Gate (Truth): expected powrush_lattice_flow_share_v1, got '{}'",
            env.schema
        ));
    }
    if env.flow_total < 0.0 || env.reserve_total < 0.0 {
        return Err("Mercy Gate (Abundance): negative flow/reserve rejected — zero-harm".into());
    }
    Ok(env)
}

/// Parse → bridging_pass → SchemaRegistry ingest.
pub fn ingest_lattice_flow_share_json(
    reg: &mut SchemaRegistry,
    json: &str,
) -> Result<BridgingPassResult, String> {
    let env = parse_powrush_lattice_flow_share_json(json)?;
    let result = bridging_pass(&env.to_bridging_context());
    reg.ingest_bridging(result.clone());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_ingest_flow_share() {
        let json = r#"{
            "schema": "powrush_lattice_flow_share_v1",
            "flow_total": 3.0,
            "reserve_total": 1.5,
            "choices_made": 4,
            "last_path": "Flow outward",
            "mercy_note": "Voluntary abundance direction — never scarcity",
            "exported_at_secs": 12.0
        }"#;
        let mut reg = SchemaRegistry::new();
        let result = ingest_lattice_flow_share_json(&mut reg, json).unwrap();
        assert!(result.high_road_effort);
        assert!(!reg.is_empty() || !result.notes.is_empty());
    }

    #[test]
    fn reject_negative_abundance() {
        let json = r#"{
            "schema": "powrush_lattice_flow_share_v1",
            "flow_total": -1.0,
            "reserve_total": 0.0,
            "choices_made": 1,
            "mercy_note": "x",
            "exported_at_secs": 1.0
        }"#;
        assert!(parse_powrush_lattice_flow_share_json(json).is_err());
    }
}
