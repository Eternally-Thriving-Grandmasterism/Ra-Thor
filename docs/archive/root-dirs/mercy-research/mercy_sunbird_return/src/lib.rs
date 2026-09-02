// mercy_sunbird_return/src/lib.rs — He³ Return Transport Swarm
#[derive(Debug, Clone)]
pub struct SunbirdCapsule {
    pub mass_kg: f64,              // dry mass 50–150 kg
    pub payload_he3_kg: f64,       // 1–10 kg
    pub sail_area_m2: f64,         // 100–500 m²
    pub transit_days: u32,         // 60–180 days
    pub valence: f64,
}

impl SunbirdCapsule {
    pub fn new() -> Self {
        SunbirdCapsule {
            mass_kg: 100.0,
            payload_he3_kg: 5.0,
            sail_area_m2: 300.0,
            transit_days: 120,
            valence: 1.0,
        }
    }

    pub fn execute_return(&self) -> bool {
        if self.valence >= 0.9999999 {
            println!("Mercy-approved: Sunbird capsule launched — {} kg He³ in transit ({} days)", 
                     self.payload_he3_kg, self.transit_days);
            true
        } else {
            println!("Mercy shield: Return aborted (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_he3_return(capsules: u32) -> f64 {
    let capsule = SunbirdCapsule::new();
    let total_he3 = capsules as f64 * capsule.payload_he3_kg;
    if capsule.execute_return() {
        println!("Total He³ return potential: {:.1} kg from {} capsules", total_he3, capsules);
        total_he3
    } else {
        0.0
    }
}
