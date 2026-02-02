// mercy_biomimetic_propulsion/src/lib.rs — Biomimetic Propulsion Selector
#[derive(Debug, Clone, PartialEq)]
pub enum PropulsionMode {
    PterosaurMorphing,
    SpinosaurusUndulation,
    SharkRibletFlow,
    GeckoPetalSwitchable,
    LadybirdVibrationDamping,
}

#[derive(Debug, Clone)]
pub struct BiomimeticPropulsion {
    pub valence: f64,
}

impl BiomimeticPropulsion {
    pub fn new() -> Self {
        BiomimeticPropulsion { valence: 1.0 }
    }

    pub fn select_mode(&self, mission_phase: &str) -> PropulsionMode {
        if self.valence < 0.9999999 {
            println!("Mercy shield: Propulsion selection paused — valence {:.7}", self.valence);
            return PropulsionMode::PterosaurMorphing; // fallback
        }

        match mission_phase {
            "cruise" => PropulsionMode::PterosaurMorphing,
            "amphibious" => PropulsionMode::SpinosaurusUndulation,
            "high-speed" => PropulsionMode::SharkRibletFlow,
            "attachment" => PropulsionMode::GeckoPetalSwitchable,
            "shock" => PropulsionMode::LadybirdVibrationDamping,
            _ => PropulsionMode::PterosaurMorphing,
        }
    }
}

pub fn simulate_propulsion_selection(phase: &str) {
    let engine = BiomimeticPropulsion::new();
    let mode = engine.select_mode(phase);
    println!("Mercy-approved: {} phase — selected {:?}", phase, mode);
}
