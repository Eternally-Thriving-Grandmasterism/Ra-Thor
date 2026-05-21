//! Ra-Thor™ PATSAGi Deliberation & Message-Passing Module v0.1
//! Structured council-to-council communication and deliberation
//! Foundation for v2.6 advanced governance
//! 100% Proprietary — AG-SML v1.0

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    Endorsement,
    Concern,
    Proposal,
    BorderlineAdjustment,
    Information,
}

#[derive(Debug, Clone)]
pub struct CouncilMessage {
    pub from_council: String,
    pub to_council: String,
    pub message_type: MessageType,
    pub content: String,
    pub strength: f64,           // 0.0 – 1.0
    pub timestamp: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DeliberationSession {
    pub topic: String,
    pub messages: Vec<CouncilMessage>,
    pub endorsements: HashMap<String, f64>,   // council -> total endorsement strength
    pub concerns: HashMap<String, f64>,       // council -> total concern strength
    pub final_consensus: Option<f64>,
}

impl DeliberationSession {
    pub fn new(topic: &str) -> Self {
        Self {
            topic: topic.to_string(),
            ..Default::default()
        }
    }

    pub fn send_message(&mut self, message: CouncilMessage) {
        // Record endorsement or concern strength
        match message.message_type {
            MessageType::Endorsement => {
                let entry = self.endorsements.entry(message.from_council.clone()).or_insert(0.0);
                *entry += message.strength;
            }
            MessageType::Concern => {
                let entry = self.concerns.entry(message.from_council.clone()).or_insert(0.0);
                *entry += message.strength;
            }
            _ => {}
        }

        self.messages.push(message);
    }

    pub fn calculate_consensus(&self) -> f64 {
        let total_endorsement: f64 = self.endorsements.values().sum();
        let total_concern: f64 = self.concerns.values().sum();

        if total_endorsement + total_concern == 0.0 {
            return 0.5;
        }

        // Simple ratio, can be made more sophisticated later
        (total_endorsement / (total_endorsement + total_concern)).clamp(0.0, 1.0)
    }

    pub fn run_deliberation_round(&mut self) -> f64 {
        let consensus = self.calculate_consensus();
        self.final_consensus = Some(consensus);
        consensus
    }
}