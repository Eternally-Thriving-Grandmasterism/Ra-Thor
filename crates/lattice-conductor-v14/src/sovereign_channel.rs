//! Sovereign Channel Prototypes — Thunder Lattice v14.0.7+
//! Foundational structures for future sovereign, mercy-gated communication channels between organisms.

use crate::distributed_mercy_mesh::MercyEvent;

/// Direction of the sovereign channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelDirection {
    Outgoing,
    Incoming,
    Bidirectional,
}

/// Status of a sovereign channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelStatus {
    Pending,
    Active,
    Suspended,
    Closed,
}

/// A prototype for a sovereign communication channel between two organisms.
#[derive(Debug, Clone)]
pub struct SovereignChannel {
    pub id: String,
    pub from_organism: String,
    pub to_organism: String,
    pub direction: ChannelDirection,
    pub status: ChannelStatus,
    pub mercy_score: f64,           // Current mercy alignment of the channel
    pub last_activity: u64,
}

impl SovereignChannel {
    pub fn new(
        from: &str,
        to: &str,
        direction: ChannelDirection,
    ) -> Self {
        Self {
            id: format!("channel_{}_{}", from, to),
            from_organism: from.to_string(),
            to_organism: to.to_string(),
            direction,
            status: ChannelStatus::Pending,
            mercy_score: 0.7,
            last_activity: 0,
        }
    }

    pub fn activate(&mut self) {
        self.status = ChannelStatus::Active;
        println!("[SOVEREIGN CHANNEL] Channel {} activated between {} and {}", self.id, self.from_organism, self.to_organism);
    }

    pub fn close(&mut self) {
        self.status = ChannelStatus::Closed;
    }

    /// Future: Send a mercy-gated message through the channel.
    pub fn send_message(&self, payload: &str) -> Option<MercyEvent> {
        if self.status == ChannelStatus::Active {
            println!("[SOVEREIGN CHANNEL] Sending message via {}: {}", self.id, payload);
            // In future this would create a proper MeshMessageReceived event
            Some(MercyEvent::MeshMessageReceived {
                from: self.from_organism.clone(),
                payload_type: "sovereign_message".to_string(),
            })
        } else {
            None
        }
    }
}

/// Simple manager for multiple sovereign channels (prototype).
pub struct SovereignChannelManager {
    channels: Vec<SovereignChannel>,
}

impl SovereignChannelManager {
    pub fn new() -> Self {
        Self { channels: Vec::new() }
    }

    pub fn create_channel(&mut self, from: &str, to: &str, direction: ChannelDirection) -> &SovereignChannel {
        let channel = SovereignChannel::new(from, to, direction);
        self.channels.push(channel);
        self.channels.last().unwrap()
    }

    pub fn get_active_channels(&self) -> Vec<&SovereignChannel> {
        self.channels.iter().filter(|c| c.status == ChannelStatus::Active).collect()
    }
}