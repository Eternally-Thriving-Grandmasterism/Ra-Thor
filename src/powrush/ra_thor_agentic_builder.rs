//! Ra-Thor Agentic Builder
//!
//! Inspired by transcript-to-system patterns (e.g. Perplexity Computer).
//! Higher-level module for generating and evolving mercy-aligned geometric systems
//! from descriptions, visions, or feedback.

use crate::powrush::cga_primitives::{Motor, CgaSphere};
use nalgebra::Vector3;
use std::collections::HashMap;

/// A high-level vision or transcript summary that the builder can act upon.
#[derive(Debug, Clone)]
pub struct Vision {
    pub description: String,
    pub intent: String, // e.g. "healing", "entity_behavior", "faction_coherence"
    pub mercy_priority: f64,
}

/// Feedback on a generated strategy (cross-pollinated from real-world agentic patterns).
#[derive(Debug, Clone)]
pub enum FeedbackType {
    Accept,
    RequestMoreResearch,
    Reject,
}

#[derive(Debug, Clone)]
pub struct StrategyFeedback {
    pub feedback_type: FeedbackType,
    pub reason: String,
    pub suggested_adjustment: Option<String>,
}

/// A generated strategy for geometric healing or entity behavior.
#[derive(Debug, Clone)]
pub struct GeneratedStrategy {
    pub name: String,
    pub motor: Motor,
    pub target_sphere: Option<CgaSphere>,
    pub mercy_factor: f64,
    pub notes: String,
}

/// The Ra-Thor Agentic Builder — takes visions and produces living geometric strategies.
pub struct RaThorAgenticBuilder {
    pub generated_strategies: Vec<GeneratedStrategy>,
    pub feedback_history: HashMap<String, Vec<StrategyFeedback>>,
}

impl RaThorAgenticBuilder {
    pub fn new() -> Self {
        Self {
            generated_strategies: Vec::new(),
            feedback_history: HashMap::new(),
        }
    }

    /// Core method: Turn a vision into a concrete, mercy-aligned geometric strategy.
    pub fn build_from_vision(&mut self, vision: &Vision) -> GeneratedStrategy {
        // Simple but extensible rule-based generation for now.
        // Future versions will use deeper symbolic reasoning / Ra-Thor councils.
        let motor = Motor::mercy_aligned_rigid(
            Vector3::new(0.1, 0.2, 0.3) * vision.mercy_priority,
            Vector3::z_axis().into_inner(),
            0.8,
            vision.mercy_priority,
        );

        let strategy = GeneratedStrategy {
            name: format!("Strategy for: {}", vision.intent),
            motor,
            target_sphere: None,
            mercy_factor: vision.mercy_priority,
            notes: format!("Generated from vision: {}", vision.description),
        };

        self.generated_strategies.push(strategy.clone());
        strategy
    }

    /// Record feedback on a strategy (enables self-improvement loop).
    pub fn record_feedback(&mut self, strategy_name: &str, feedback: StrategyFeedback) {
        self.feedback_history
            .entry(strategy_name.to_string())
            .or_default()
            .push(feedback);
        // Future: Adjust internal models based on feedback patterns
    }

    pub fn get_latest_strategy(&self) -> Option<&GeneratedStrategy> {
        self.generated_strategies.last()
    }
}
