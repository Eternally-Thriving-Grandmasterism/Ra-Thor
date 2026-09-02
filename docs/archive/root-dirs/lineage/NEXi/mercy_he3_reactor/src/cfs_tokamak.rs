// mercy_he3_reactor/src/cfs_tokamak.rs — Mercy-Gated CFS Tokamak Confinement
#[derive(Debug, Clone)]
pub struct CFS_Tokamak {
    pub field_strength_t: f64,      // 18–20 T
    pub q_target: f64,              // >2–10
    pub power_mw: f64,
    pub valence: f64,
}

impl CFS_Tokamak {
    pub fn new(power_mw: f64) -> Self {
        CFS_Tokamak {
            field_strength_t: 18.0,
            q_target: 5.0,
            power_mw,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            println!("Mercy-approved: CFS tokamak online — {} MW fusion (field {:.1} T, Q target {:.1})", 
                     self.power_mw, self.field_strength_t, self.q_target);
            true
        } else {
            println!("Mercy shield: CFS tokamak rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_cfs_tokamak(power_mw: f64) -> bool {
    let reactor = CFS_Tokamak::new(power_mw);
    reactor.operate()
}
