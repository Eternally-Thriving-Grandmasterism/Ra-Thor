// mercy_os_kernel/src/valence_gate.rs — Core Mercy Gate
#[derive(Debug, Clone)]
pub struct ValenceGate {
    pub current_valence: f64,
    pub threshold: f64,
}

impl ValenceGate {
    pub fn new() -> Self {
        ValenceGate {
            current_valence: 1.0,
            threshold: 0.9999999,
        }
    }

    pub fn allow(&self, operation: &str) -> bool {
        if self.current_valence >= self.threshold {
            println!("Mercy-approved: {} permitted", operation);
            true
        } else {
            println!("Mercy shield: {} rejected — valence {:.7}", operation, self.current_valence);
            false
        }
    }

    pub fn update(&mut self, delta: f64) {
        self.current_valence = (self.current_valence + delta).clamp(0.0, 1.0);
        println!("Valence updated to: {:.7}", self.current_valence);
    }
}
