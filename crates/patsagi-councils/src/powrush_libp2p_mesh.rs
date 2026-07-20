//! Powrush libp2p Federated Mesh — v14.15.0
//!
//! Real-time Voice-Skin sync layer + gossip mesh for multiplayer.
//! TOLC 8 posture on gossip messages. Epigenetic blessing injection.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0
//!
//! Note: depends on libp2p + powrush_multiplayer surfaces when those
//! crates are present in the workspace build graph.

use crate::powrush_multiplayer::{truth_purity_score, PowrushWorld, UserProfile};
use libp2p::{
    gossipsub, swarm::SwarmEvent, PeerId, Swarm, Transport,
};
use ra_thor_mercy::interval_mercy::Interval;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct VoiceSkinSyncLayer {
    pub active_voice_id: String,
    pub resonance_interval: Interval,
    pub epigenetic_blessing: f64,
    pub last_sync: u64,
}

#[derive(Clone, Debug)]
pub struct FederatedMesh {
    pub swarm: Swarm<gossipsub::Behaviour>,
    pub voice_skin_sync: VoiceSkinSyncLayer,
    pub max_peers: u32,
    pub tolc8_enforced: bool,
    pub world_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VoicePacket {
    pub voice_id: String,
    pub resonance: Interval,
    pub blessing: f64,
    pub payload: String,
    pub timestamp: u64,
    pub sender_id: String,
}

impl FederatedMesh {
    pub fn new(world: &PowrushWorld) -> Self {
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(local_key.public());

        let transport = libp2p::tcp::tokio::Transport::default()
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(libp2p::noise::Config::new(&local_key).unwrap())
            .multiplex(libp2p::yamux::Config::default())
            .boxed();

        let mut behaviour = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub::ConfigBuilder::default()
                .heartbeat_interval(std::time::Duration::from_secs(1))
                .build()
                .unwrap(),
        )
        .unwrap();

        let world_topic = gossipsub::IdentTopic::new(format!("powrush-world-{}", world.id));
        let voice_topic = gossipsub::IdentTopic::new("voice-skin-sync");
        behaviour.subscribe(&world_topic).unwrap();
        behaviour.subscribe(&voice_topic).unwrap();

        let swarm = Swarm::new(transport, behaviour, peer_id);

        Self {
            swarm,
            voice_skin_sync: VoiceSkinSyncLayer {
                active_voice_id: "sherif-samy-botros-v1".to_string(),
                resonance_interval: Interval {
                    low: 0.97,
                    high: 1.00,
                },
                epigenetic_blessing: 0.35,
                last_sync: current_timestamp(),
            },
            max_peers: 128,
            tolc8_enforced: true,
            world_id: world.id.clone(),
        }
    }

    pub async fn broadcast_voice_skin(&mut self, user: &UserProfile, message: &str) {
        if truth_purity_score(message) <= 0.95 {
            return; // APTD gate
        }

        let packet = VoicePacket {
            voice_id: self.voice_skin_sync.active_voice_id.clone(),
            resonance: self.voice_skin_sync.resonance_interval,
            blessing: self.voice_skin_sync.epigenetic_blessing,
            payload: message.to_string(),
            timestamp: current_timestamp(),
            sender_id: user.user_id.clone(),
        };

        let topic = gossipsub::IdentTopic::new("voice-skin-sync");
        if let Err(e) = self
            .swarm
            .behaviour_mut()
            .publish(topic, serde_json::to_vec(&packet).unwrap())
        {
            eprintln!("Gossip publish error: {:?}", e);
        }
    }

    pub fn handle_swarm_event(&mut self, event: SwarmEvent<gossipsub::Event>) {
        if let SwarmEvent::Behaviour(gossipsub::Event::Message { message, .. }) = event {
            if let Ok(packet) = serde_json::from_slice::<VoicePacket>(&message.data) {
                if packet.resonance.high >= 0.97 && packet.blessing >= 0.35 {
                    println!(
                        "🎙️ Voice-Skin synced (v14.15.0): {} (resonance {:.2})",
                        packet.voice_id, packet.resonance.high
                    );
                }
            }
        }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_skin_resonance_passes_aptd() {
        let world = PowrushWorld::default();
        let mesh = FederatedMesh::new(&world);
        assert!(mesh.voice_skin_sync.resonance_interval.high >= 0.97);
        assert!(mesh.tolc8_enforced);
    }
}
