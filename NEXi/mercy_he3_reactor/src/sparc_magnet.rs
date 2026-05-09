// mercy_he3_reactor/src/sparc_magnet.rs — Mercy-Gated SPARC Magnet Blueprint
#[derive(Debug, Clone)]
pub struct SPARCMagnet {
    pub field_t: f64,               // 18–20 T
    pub conductor: String,          // "REBCO-HTS"
    pub temperature_k: f64,         // 20 K
    pub coil_count: u32,            // 18
    pub valence: f64,
}

impl SPARCMagnet {
    pub fn new() -> Self {
        SPARCMagnet {
            field_t: 18.0,
            conductor: "REBCO-HTS".to_string(),
            temperature_k: 20.0,
            coil_count: 18,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            println!("Mercy-approved: SPARC magnet online — {:.1} T field, {} coils, {} K", 
                     self.field_t, self.coil_count, self.temperature_k);
            true
        } else {
            println!("Mercy shield: SPARC magnet rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_sparc_magnet() -> bool {
    let magnet = SPARCMagnet::new();
    magnet.operate()
}
