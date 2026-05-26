// src/powrush/geometric_self_healing.rs
// v14.0.9 Powrush RBE + Geometric Self-Healing
//
// This module introduces geometric self-healing mechanics for the Powrush RBE economy
// using multivector representations and sandwich product transformations.
//
// Status: High-quality prototype / seed module
// Future: Deeper integration with actual Powrush faction systems, full CGA substrate,
//         player organism coherence as CGA entities, and reward vector healing.

use std::time::SystemTime;

/// Represents a resource or motivation vector in the RBE economy as a multivector.
#[derive(Debug, Clone)]
pub struct ResourceVector {
    pub scalar: f64,
    pub vector: [f64; 3],
    pub bivector: [f64; 3],
    pub last_update: SystemTime,
}

impl ResourceVector {
    pub fn new(scalar: f64, vector: [f64; 3]) -> Self {
        Self {
            scalar,
            vector,
            bivector: [0.0, 0.0, 0.0],
            last_update: SystemTime::now(),
        }
    }

    /// Applies a mercy-aligned transformation using a simplified sandwich product.
    /// Note: This is a linear approximation for prototype purposes.
    /// True geometric algebra sandwich product (R * M * ~R) to be implemented with proper CGA crate.
    pub fn sandwich_transform(&mut self, motor: &CliffordMotor) {
        let s = motor.scale;
        self.scalar *= s;
        for i in 0..3 {
            self.vector[i] = self.vector[i] * s + motor.rotation[i] * 0.1;
            self.bivector[i] *= s * 0.95;
        }
        self.last_update = SystemTime::now();
    }
}

/// A simplified motor (versor) used to apply mercy-aligned transformations.
#[derive(Debug, Clone)]
pub struct CliffordMotor {
    pub scale: f64,
    pub rotation: [f64; 3],
    pub mercy_alignment: f64,
}

impl CliffordMotor {
    pub fn mercy_aligned(strength: f64, alignment: f64) -> Self {
        Self {
            scale: 1.0 + strength * alignment,
            rotation: [
                strength * 0.3 * alignment,
                strength * 0.2 * alignment,
                strength * 0.1 * alignment,
            ],
            mercy_alignment: alignment,
        }
    }
}

/// Geometric healing field for Powrush RBE state.
#[derive(Debug, Clone)]
pub struct PowrushHealingField {
    pub resource_flow_coherence: f64,
    pub faction_harmony: f64,
    pub motivation_coherence: f64,
    pub active_vectors: Vec<ResourceVector>,
}

impl PowrushHealingField {
    pub fn new() -> Self {
        Self {
            resource_flow_coherence: 0.91,
            faction_harmony: 0.87,
            motivation_coherence: 0.94,
            active_vectors: vec![ResourceVector::new(120.0, [45.0, 30.0, 15.0])],
        }
    }

    /// Monitors coherence and applies geometric healing when anomalies are detected.
    pub fn monitor_and_heal(&mut self, reason: &str) {
        if self.resource_flow_coherence < 0.75 || self.faction_harmony < 0.70 {
            let motor = CliffordMotor::mercy_aligned(0.35, 0.92);
            for v in &mut self.active_vectors {
                v.sandwich_transform(&motor);
            }
            self.resource_flow_coherence = (self.resource_flow_coherence + 0.12).min(1.0);
            self.faction_harmony = (self.faction_harmony + 0.09).min(1.0);
            println!("[Powrush] Healed RBE anomaly via sandwich transform: {}", reason);
        }
    }
}

/// Integration hook with Watchdog Thread graded responses.
pub fn integrate_with_watchdog(powrush: &mut PowrushHealingField) {
    if powrush.motivation_coherence < 0.75 {
        println!("[Integration] Escalating low motivation to Watchdog Level 3");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandwich_transform_increases_scalar() {
        let mut vec = ResourceVector::new(100.0, [10.0, 20.0, 5.0]);
        let motor = CliffordMotor::mercy_aligned(0.4, 0.95);
        vec.sandwich_transform(&motor);
        assert!(vec.scalar > 100.0);
    }

    #[test]
    fn test_monitor_and_heal_restores_coherence() {
        let mut field = PowrushHealingField::new();
        field.resource_flow_coherence = 0.60; // force anomaly
        field.monitor_and_heal("test_anomaly");
        assert!(field.resource_flow_coherence > 0.70);
    }
}