/* NOTE: This is a minimal append for the new prototype.
   In real scenario we would do a proper edit of the full file.
   For now we add the new demo function at the end. */

use crate::cga_primitives::{Motor, CgaPoint};

impl CliffordHealingField {
    /// Prototype: Real Motor sandwich-style healing step.
    /// When full-clifford feature matures, this becomes true geometric (M * P * ~M) healing.
    pub fn demo_cga_motor_healing_step(&mut self, mercy: f64) {
        if mercy < 0.6 { return; }
        let motor = Motor::from_rotation_translation(
            [1.0, 0.0, 0.0, 0.0],
            nalgebra::Vector3::new(mercy * 0.08, 0.0, 0.0)
        );
        for field in self.organism_fields.values_mut() {
            let p = CgaPoint::from_vector(field.alignment);
            let transformed = motor.sandwich_transform(p);
            field.alignment = transformed.to_vector();
            field.mercy = (field.mercy + mercy * 0.04).min(1.0);
        }
        self.mercy_flow = (self.mercy_flow + mercy * 0.02).min(1.0);
    }
}

#[cfg(test)]
mod cga_motor_prototype_tests {
    use super::*;
    #[test]
    fn cga_motor_healing_runs() {
        let mut f = CliffordHealingField::new("CGA Prototype Test");
        f.add_organism(42, nalgebra::Vector3::zeros(), nalgebra::Vector3::zeros(), nalgebra::Vector3::new(1.0,0.0,0.0), 0.85);
        f.demo_cga_motor_healing_step(0.95);
        assert!(f.mercy_flow > 0.85);
    }
}