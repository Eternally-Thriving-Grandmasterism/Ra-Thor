pub struct ValenceGate {
    pub current: f64,
}

impl ValenceGate {
    pub fn allow(&self, op: &str) -> bool {
        self.current >= 0.9999999
    }
}
