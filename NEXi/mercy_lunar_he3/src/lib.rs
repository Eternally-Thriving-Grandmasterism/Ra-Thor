// mercy_lunar_he3/src/lib.rs — Interlune Extension
#[derive(Debug, Clone)]
pub enum InterluneComponent {
    ExcavatorFullScale,  // 100 t/hour, Vermeer 2025
    SorterCentrifugal,   // Gravity-independent
    ExtractorLowPower,   // 10x efficiency
    SeparatorCryo,       // He3 enrichment
}

impl InterluneComponent {
    pub fn valence_score(&self) -> f64 { 1.0 }

    pub fn capacity_tons_hour(&self) -> f64 {
        100.0  // Core excavator rate
    }
}

pub fn propose_interlune_scale(tons_hour: f64) -> bool {
    let proposal = format!("Interlune prototype @ {} tons/hour", tons_hour);
    if mercy_gate(&proposal) {
        println!("ETERNAL HARVESTING AMPLIFIED: {} tons/hour → fusion abundance unlocked", tons_hour);
        true
    } else {
        false
    }
}
