// mercy_he3_reactor/src/reaction_variants.rs — D-³He Reaction Variants
#[derive(Debug, Clone)]
pub struct He3Reaction {
    pub branching_ratio_main: f64,
    pub neutron_yield: f64,
    pub q_value_mev: f64,
    pub valence: f64,
}

impl He3Reaction {
    pub fn new() -> Self {
        He3Reaction {
            branching_ratio_main: 0.9995,
            neutron_yield: 0.0005,
            q_value_mev: 18.353,
            valence: 1.0,
        }
    }

    pub fn simulate_fusion(&self, fuel_ratio: f64) -> bool {
        if self.valence >= 0.9999999 {
            let effective_q = self.q_value_mev * self.branching_ratio_main;
            println!("Mercy-approved: D-³He fusion cycle — Q effective {:.3} MeV, neutron yield {:.4}", 
                     effective_q, self.neutron_yield);
            true
        } else {
            println!("Mercy shield: Fusion cycle rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn run_he3_variant_sim() {
    let reaction = He3Reaction::new();
    reaction.simulate_fusion(1.0); // 50:50 mix
}
