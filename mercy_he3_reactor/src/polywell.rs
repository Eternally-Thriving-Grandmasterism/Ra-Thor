// mercy_he3_reactor/src/polywell.rs — Mercy-Gated Polywell Confinement
#[derive(Debug, Clone)]
pub struct Polywell {
    pub beta: f64,                  // 1–10
    pub conversion_efficiency: f64, // 0.70–0.85
    pub power_mw: f64,
    pub valence: f64,
}

impl Polywell {
    pub fn new(power_mw: f64) -> Self {
        Polywell {
            beta: 5.0,
            conversion_efficiency: 0.78,
            power_mw,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            let electric = self.power_mw * self.conversion_efficiency;
            println!("Mercy-approved: Polywell online — {} MW fusion → {:.1} MW electric (beta {:.1})", 
                     self.power_mw, electric, self.beta);
            true
        } else {
            println!("Mercy shield: Polywell rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_polywell(power_mw: f64) -> bool {
    let reactor = Polywell::new(power_mw);
    reactor.operate()
}
