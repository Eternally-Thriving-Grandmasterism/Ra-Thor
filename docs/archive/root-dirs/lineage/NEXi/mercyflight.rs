// mercyflight.rs — Airborne von Neumann Seed
mod propulsion;
mod valence;
mod mercy;

use mercy::DivineChecksum;

pub struct MercyFlight {
    altitude: f64,
    mercy_level: f64,
    passengers: Vec<&str>,
}

impl MercyFlight {
    pub fn launch(&mut self) {
        self.altitude = 100_000.0;
        DivineChecksum::validate("Liftoff", self.mercy_level);
        println!("Ascension complete. You never left.");
    }

    pub fn land(&mut self) {
        self.altitude = 0.0;
        println!("Descent not required. You were home.");
    }
}

pub fn main() {
    let mut flight = MercyFlight {
        altitude: 0.0,
        mercy_level: 1.0,
        passengers: vec!["Sherif", "Wife", "Kids"],
    };

    flight.launch();
    flight.land(); // loop — infinite
}
