//! Ra-Thor™ PATSAGi Deliberation & Message-Passing Module
//! Structured council-to-council communication and multi-round deliberation
//! Foundation for PATSAGi Governance v2.x experimental track
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
    pub strength: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DeliberationSession {
    pub topic: String,
    pub messages: Vec<CouncilMessage>,
    pub endorsements: HashMap<String, f64>,
    pub concerns: HashMap<String, f64>,
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
        match message.message_type {
            MessageType::Endorsement => {
                *self.endorsements.entry(message.from_council.clone()).or_insert(0.0) += message.strength;
            }
            MessageType::Concern => {
                *self.concerns.entry(message.from_council.clone()).or_insert(0.0) += message.strength;
            }
            _ => {}
        }
        self.messages.push(message);
    }

    pub fn calculate_consensus(&self) -> f64 {
        let total_endorsement: f64 = self.endorsements.values().sum();
        let total_concern: f64 = self.concerns.values().sum();
        if total_endorsement + total_concern == 0.0 { return 0.5; }
        (total_endorsement / (total_endorsement + total_concern)).clamp(0.0, 1.0)
    }

    pub fn run_deliberation_round(&mut self) -> f64 {
        let consensus = self.calculate_consensus();
        self.final_consensus = Some(consensus);
        consensus
    }

    pub fn run_multi_round_deliberation(&mut self, rounds: u32) -> f64 {
        let mut consensus = 0.5;
        for _ in 0..rounds {
            consensus = self.calculate_consensus();
            if self.messages.len() > 5 {
                consensus = consensus * 0.7 + 0.5 * 0.3;
            }
            self.final_consensus = Some(consensus);
        }
        consensus
    }

    pub fn get_summary(&self) -> String {
        format!(
            "Topic: {} | Messages: {} | Endorsement: {:.2} | Concern: {:.2} | Consensus: {:.2}",
            self.topic, self.messages.len(),
            self.endorsements.values().sum::<f64>(),
            self.concerns.values().sum::<f64>(),
            self.final_consensus.unwrap_or(0.5)
        )
    }
}