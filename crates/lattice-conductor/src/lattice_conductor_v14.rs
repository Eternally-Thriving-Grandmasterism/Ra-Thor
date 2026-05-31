// crates/lattice-conductor/src/lattice_conductor_v14.rs
// Lattice Conductor v14.4 — Real Estate Lattice + Geometric Intelligence
// Now powered by geometric-intelligence crate

use std::collections::HashMap;
use std::fmt;

use geometric_intelligence::{compute_geometric_harmony, GeometricHarmonyScore};

/// TOLC 8 Living Mercy Gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyGate {
    Truth,
    Order,
    Love,
    Compassion,
    Service,
    Abundance,
    Joy,
    CosmicHarmony,
}

impl fmt::Display for MercyGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MercyGate::Truth => write!(f, "Truth (APTD)"),
            MercyGate::Order => write!(f, "Order (Structural Harmony)"),
            MercyGate::Love => write!(f, "Love (Positive Propagation)"),
            MercyGate::Compassion => write!(f, "Compassion (Zero-Harm Reroute)"),
            MercyGate::Service => write!(f, "Service (Co-Creation)"),
            MercyGate::Abundance => write!(f, "Abundance (Mercy-Gated Flow)"),
            MercyGate::Joy => write!(f, "Joy (Valence Growth)"),
            MercyGate::CosmicHarmony => write!(f, "Cosmic Harmony (Inter-Council Sync)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Valence(f64);

impl Valence {
    pub const MIN: f64 = 0.9999999;
    pub const MAX: f64 = 1.0;

    pub fn new(v: f64) -> Result<Self, ConductorError> {
        if v >= Self::MIN && v <= Self::MAX {
            Ok(Valence(v))
        } else {
            Err(ConductorError::ValenceOutOfRange(v))
        }
    }

    pub fn value(&self) -> f64 { self.0 }
    pub fn passes_mercy(&self) -> bool { self.0 >= Self::MIN }
}

#[derive(Debug, Clone)]
pub struct RealEstateOffer {
    pub id: String,
    pub address: String,
    pub price: f64,
    pub jurisdiction: String,
    pub regulatory_flags: Vec<String>,
    pub attom_enriched: bool,
    pub base_valence: Valence,
}

#[derive(Debug, Clone)]
pub struct AttomData {
    pub property_id: String,
    pub tax_history: Vec<f64>,
    pub ownership_changes: u32,
    pub risk_score: f64,
}

#[derive(Debug, Clone)]
pub struct ConductedOffer {
    pub offer: RealEstateOffer,
    pub final_valence: Valence,
    pub mercy_gates_passed: Vec<MercyGate>,
    pub attom_snapshot: Option<AttomData>,
    pub regulatory_cleared: bool,
    pub geometric_harmony_multiplier: f64,
    pub geometric_resonance_notes: String,
    pub conducted_at: u64,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ConductorError {
    #[error("Valence {0} out of mercy range")]
    ValenceOutOfRange(f64),
    #[error("Mercy gate {gate} failed: {reason}")]
    MercyGateFailed { gate: MercyGate, reason: String },
    #[error("Regulatory block: {0}")]
    RegulatoryBlock(String),
    #[error("Lattice conduction invariant broken: {0}")]
    InvariantBroken(String),
}

pub struct LatticeConductor {
    pub version: &'static str,
    mercy_gates: Vec<MercyGate>,
    attom_cache: HashMap<String, AttomData>,
    regulatory_rules: HashMap<String, String>,
}

impl Default for LatticeConductor {
    fn default() -> Self { Self::new() }
}

impl LatticeConductor {
    pub fn new() -> Self {
        let gates = vec![
            MercyGate::Truth, MercyGate::Order, MercyGate::Love, MercyGate::Compassion,
            MercyGate::Service, MercyGate::Abundance, MercyGate::Joy, MercyGate::CosmicHarmony,
        ];
        let mut rules = HashMap::new();
        rules.insert("Ontario".to_string(), "RESA/TRESA compliance + reverse onus safety checks".to_string());
        rules.insert("USA".to_string(), "State-level disclosure + federal fair housing".to_string());

        LatticeConductor {
            version: "v14.4.0-geometric-intelligence",
            mercy_gates: gates,
            attom_cache: HashMap::new(),
            regulatory_rules: rules,
        }
    }

    pub fn enforce_mercy_gates(&self, offer: &RealEstateOffer) -> Result<Vec<MercyGate>, ConductorError> {
        let mut passed = Vec::new();
        for gate in &self.mercy_gates {
            match gate {
                MercyGate::Truth => {
                    if offer.base_valence.passes_mercy() { passed.push(*gate); }
                    else { return Err(ConductorError::MercyGateFailed { gate: *gate, reason: "Base valence below mercy threshold".into() }); }
                }
                MercyGate::Compassion => {
                    if offer.regulatory_flags.iter().any(|f| f.contains("harm")) {
                        return Err(ConductorError::MercyGateFailed { gate: *gate, reason: "Potential harm flag detected".into() });
                    }
                    passed.push(*gate);
                }
                MercyGate::Order => {
                    if offer.jurisdiction.is_empty() {
                        return Err(ConductorError::MercyGateFailed { gate: *gate, reason: "Missing jurisdiction".into() });
                    }
                    passed.push(*gate);
                }
                _ => passed.push(*gate),
            }
        }
        Ok(passed)
    }

    pub fn integrate_attom(&mut self, offer: &RealEstateOffer) -> Result<AttomData, ConductorError> {
        if let Some(cached) = self.attom_cache.get(&offer.id) { return Ok(cached.clone()); }
        let data = AttomData { property_id: format!("ATTOM-{}", offer.id), tax_history: vec![], ownership_changes: 0, risk_score: 0.1 };
        self.attom_cache.insert(offer.id.clone(), data.clone());
        Ok(data)
    }

    pub fn check_regulatory(&self, offer: &RealEstateOffer) -> Result<bool, ConductorError> {
        Ok(true)
    }

    /// Uses the new geometric-intelligence crate
    pub fn compute_geometric_harmony(&self, offer: &RealEstateOffer) -> GeometricHarmonyScore {
        let tolc_proxy: u32 = if offer.price > 1_000_000.0 { 89 } else { 34 };
        compute_geometric_harmony(tolc_proxy, offer.base_valence.value())
    }

    pub fn conduct_real_estate_offer(&mut self, offer: RealEstateOffer) -> Result<ConductedOffer, ConductorError> {
        let passed_gates = self.enforce_mercy_gates(&offer)?;
        let attom = if offer.attom_enriched { Some(self.integrate_attom(&offer)?) } else { None };
        let cleared = self.check_regulatory(&offer)?;

        // Use new geometric intelligence layer
        let harmony = self.compute_geometric_harmony(&offer);

        let final_valence = offer.base_valence;

        Ok(ConductedOffer {
            offer,
            final_valence,
            mercy_gates_passed: passed_gates,
            attom_snapshot: attom,
            regulatory_cleared: cleared,
            geometric_harmony_multiplier: harmony.multiplier,
            geometric_resonance_notes: harmony.resonance_notes,
            conducted_at: 1748628960,
        })
    }

    pub fn conduct_batch(&mut self, offers: Vec<RealEstateOffer>) -> Vec<Result<ConductedOffer, ConductorError>> {
        offers.into_iter().map(|o| self.conduct_real_estate_offer(o)).collect()
    }
}
