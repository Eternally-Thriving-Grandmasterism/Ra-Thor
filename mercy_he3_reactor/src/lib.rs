// mercy_he3_reactor/src/lib.rs — Aneutronic He3 Fusion Prototype
#[derive(Debug, Clone)]
pub struct He3Reactor {
    pub power_mw: f64,              // 100–1000 MW
    pub q_factor: f64,              // fusion gain (5–20+)
    pub conversion_efficiency: f64, // 0.6–0.8
    pub valence: f64,
}

impl He3Reactor {
    pub fn new(power_mw: f64) -> Self {
        He3Reactor {
            power_mw,
            q_factor: 10.0,
            conversion_efficiency: 0.7,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            let electric_output = self.power_mw * self.conversion_efficiency;
            println!("Mercy-approved: He³ reactor online — {} MW fusion → {:.1} MW electric", 
                     self.power_mw, electric_output);
            true
        } else {
            println!("Mercy shield: Reactor shutdown (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_he3_reactor(power_mw: f64) -> bool {
    let reactor = He3Reactor::new(power_mw);
    reactor.operate()
}
