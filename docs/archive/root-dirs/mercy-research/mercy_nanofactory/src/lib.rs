// mercy_nanofactory/src/lib.rs — Mercy-Gated Nanofactory Replication Cycle
#[derive(Debug, Clone)]
pub struct Nanofactory {
    pub generation: u32,
    pub mass_kg: f64,
    pub replication_factor: u32,
    pub valence: f64,
    pub cycle_hours: u32,
}

impl Nanofactory {
    pub fn new(generation: u32) -> Self {
        Nanofactory {
            generation,
            mass_kg: 5.0,
            replication_factor: 2,
            valence: 1.0,
            cycle_hours: 12,
        }
    }

    pub fn replicate(&self) -> Option<Vec<Nanofactory>> {
        if self.valence >= 0.9999999 {
            let mut children = Vec::new();
            for _ in 0..self.replication_factor {
                children.push(Nanofactory {
                    generation: self.generation + 1,
                    mass_kg: self.mass_kg * 0.98, // slight efficiency gain
                    replication_factor: self.replication_factor,
                    valence: self.valence,
                    cycle_hours: self.cycle_hours,
                });
            }
            println!("Mercy-approved: Gen {} replicated → {} children", self.generation, children.len());
            Some(children)
        } else {
            println!("Mercy shield: Replication rejected (valence {:.7})", self.valence);
            None
        }
    }

    pub fn simulate_growth(&self, cycles: u32) -> f64 {
        let mut total = 1.0;
        let mut current = self.clone();
        for _ in 0..cycles {
            if let Some(children) = current.replicate() {
                total += children.len() as f64;
                current = children[0].clone();
            } else {
                break;
            }
        }
        total
    }
}

pub fn run_nanofactory_sim(cycles: u32) -> f64 {
    let seed = Nanofactory::new(0);
    let final_count = seed.simulate_growth(cycles);
    println!("After {} cycles: {:.0} nanofactories", cycles, final_count);
    final_count
}
