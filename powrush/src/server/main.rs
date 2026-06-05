use crate::multi_agent_orchestrator::{RichAgentState, MoralEvaluation, NeuralQNetwork};

// Example extension in game state snapshot
#[derive(Serialize)]
pub struct EnrichedNpcState {
    pub entity_id: u64,
    pub goal: String,
    pub emotional_state: EmotionalState,
    pub rich_state: RichAgentState,
    pub q_values: Option<QValues>,
    pub moral_evaluation: Option<MoralEvaluation>,
    pub combined_wisdom_score: f32,
}

// In the WebSocket / DataChannel snapshot logic, include EnrichedNpcState for connected clients.