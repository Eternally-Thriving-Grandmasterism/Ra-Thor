//! adaptive_guide.rs
//! Soft adaptive learning for human participants (~90-second window)
//! Phase C — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

/// Preferred pace / curiosity style learned during first-contact window
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GuideStyle {
    Gentle,
    Curious,
    Direct,
    Exploratory,
    Balanced, // default until learned
}

/// Tracks the short adaptive learning window for a human participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveGuide {
    pub participant_id: String,
    pub started_at: DateTime<Utc>,
    pub current_style: GuideStyle,
    pub interaction_count: u32,
}

impl AdaptiveGuide {
    pub fn new(participant_id: impl Into<String>) -> Self {
        Self {
            participant_id: participant_id.into(),
            started_at: Utc::now(),
            current_style: GuideStyle::Balanced,
            interaction_count: 0,
        }
    }

    /// Returns true if still inside the ~90-second learning window
    pub fn is_learning(&self) -> bool {
        Utc::now() - self.started_at < Duration::seconds(90)
    }

    /// Soft update of preferred style based on simple interaction signals
    pub fn observe_interaction(&mut self, signal: &str) {
        self.interaction_count += 1;

        if !self.is_learning() {
            return;
        }

        // Extremely lightweight heuristic — will later bind to real soft-play signals
        self.current_style = match signal {
            "slow" | "pause" | "reflect" => GuideStyle::Gentle,
            "ask" | "why" | "curious" => GuideStyle::Curious,
            "go" | "next" | "fast" => GuideStyle::Direct,
            "explore" | "look" | "wander" => GuideStyle::Exploratory,
            _ => GuideStyle::Balanced,
        };
    }

    pub fn style(&self) -> &GuideStyle {
        &self.current_style
    }
}
