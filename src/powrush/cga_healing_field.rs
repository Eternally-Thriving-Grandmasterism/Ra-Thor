//! CgaHealingField — Geometric Healing System for CGA Entities
//!
//! Manages and applies mercy-aligned Conformal Geometric Algebra healing
//! to multiple CgaEntity instances, with support for intersection-based healing.

use crate::powrush::cga_entity::CgaEntity;
use crate::powrush::cga_primitives::{Motor, CgaSphere};
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct CgaHealingField {
    pub name: String,
    pub entities: Vec<CgaEntity>,
}

impl CgaHealingField {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entities: Vec::new(),
        }
    }

    pub fn add_entity(&mut self, entity: CgaEntity) {
        self.entities.push(entity);
    }

    pub fn apply_motor_to_all(&mut self, motor: &Motor) {
        for entity in &mut self.entities {
            entity.apply_motor(motor);
        }
    }

    /// Applies healing to all entities that intersect with the given sphere.
    pub fn heal_entities_in_sphere(
        &mut self,
        healing_sphere: &CgaSphere,
        healing_motor: &Motor,
        mercy: f64,
        progress: f64,
    ) {
        for entity in &mut self.entities {
            if entity.intersects_sphere(healing_sphere) {
                entity.smooth_heal(healing_motor, progress, mercy);
            }
        }
    }

    pub fn apply_batch_healing(
        &mut self,
        healing_direction: Vector3<f64>,
        strength: f64,
        mercy: f64,
        progress: f64,
    ) {
        let healing_motor = Motor::mercy_aligned_rigid(
            healing_direction,
            Vector3::new(0.0, 0.0, 1.0),
            strength,
            mercy,
        );

        for entity in &mut self.entities {
            entity.smooth_heal(&healing_motor, progress, mercy);
        }
    }

    pub fn heal_entity_by_id(
        &mut self,
        entity_id: u64,
        healing_motor: &Motor,
        mercy: f64,
        progress: f64,
    ) -> bool {
        if let Some(entity) = self.entities.iter_mut().find(|e| e.id == entity_id) {
            entity.smooth_heal(healing_motor, progress, mercy);
            true
        } else {
            false
        }
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}
