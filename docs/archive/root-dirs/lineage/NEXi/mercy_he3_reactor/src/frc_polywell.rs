// mercy_he3_reactor/src/frc_polywell.rs — Mercy-Gated FRC-Polywell Hybrid
#[derive(Debug, Clone)]
pub struct FRCPolywell {
    pub beta: f64,                  // 0.8–1.0
    pub conversion_efficiency: f64, // 0.50–0.85
    pub power_mw: f64,
    pub valence: f64,
}

impl FRCPolywell {
    pub fn new(power_mw: f64) -> Self {
        FRCPolywell {
            beta: 0.95,
            conversion_efficiency: 0.72,
            power_mw,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            let electric = self.power_mw * self.conversion_efficiency;
            println!("Mercy-approved: FRC-Polywell online — {} MW fusion → {:.1} MW electric (beta {:.2})", 
                     self.power_mw, electric, self.beta);
            true
        } else {
            println!("Mercy shield: FRC-Polywell rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_frc_polywell(power_mw: f64) -> bool {
    let reactor = FRCPolywell::new(power_mw);
    reactor.operate()
}
