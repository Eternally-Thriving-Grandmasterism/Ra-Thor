//! crates/powrush/src/npc/behavior.rs
//! Hybrid Behavior Tree for Powrush NPCs — mercy-gated via MIAL/MWPO

use super::NpcAgent;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeStatus {
    Success,
    Failure,
    Running,
}

/// Core trait for any behavior node (extensible)
pub trait BehaviorNode {
    fn tick(&mut self, agent: &mut NpcAgent) -> NodeStatus;
}

/// Simple enum-based tree for performance + composability
#[derive(Debug)]
pub enum BehaviorTree {
    Selector(Vec<Box<dyn BehaviorNode>>),
    Sequence(Vec<Box<dyn BehaviorNode>>),
    MercyGateDecorator {
        min_valence: f64,
        child: Box<dyn BehaviorNode>,
    },
    Leaf(LeafBehavior),
}

#[derive(Debug)]
pub enum LeafBehavior {
    AggressiveAttack,
    DefensiveRetreat,
    PassiveObserve,
    AwakenedCooperate,
    DesperateFlee,
    Patrol,
    // Add more as needed
}

impl BehaviorNode for BehaviorTree {
    fn tick(&mut self, agent: &mut NpcAgent) -> NodeStatus {
        match self {
            BehaviorTree::Selector(children) => {
                for child in children {
                    match child.tick(agent) {
                        NodeStatus::Success => return NodeStatus::Success,
                        NodeStatus::Running => return NodeStatus::Running,
                        _ => continue,
                    }
                }
                NodeStatus::Failure
            }
            BehaviorTree::Sequence(children) => {
                for child in children {
                    match child.tick(agent) {
                        NodeStatus::Failure => return NodeStatus::Failure,
                        NodeStatus::Running => return NodeStatus::Running,
                        _ => continue,
                    }
                }
                NodeStatus::Success
            }
            BehaviorTree::MercyGateDecorator { min_valence, child } => {
                // MIAL/MWPO mercy check
                if agent.stats.mercy_valence >= *min_valence {
                    child.tick(agent)
                } else {
                    NodeStatus::Failure
                }
            }
            BehaviorTree::Leaf(leaf) => {
                // Placeholder leaf execution — expand with real logic
                match leaf {
                    LeafBehavior::Patrol => NodeStatus::Success,
                    _ => NodeStatus::Running,
                }
            }
        }
    }
}

impl BehaviorTree {
    pub fn default_mercy_aware() -> Self {
        BehaviorTree::Selector(vec![
            Box::new(BehaviorTree::MercyGateDecorator {
                min_valence: 0.85,
                child: Box::new(BehaviorTree::Leaf(LeafBehavior::AwakenedCooperate)),
            }),
            Box::new(BehaviorTree::Leaf(LeafBehavior::Patrol)),
        ])
    }

    pub fn apply_mercy_refinement(&mut self) {
        // MWPO-triggered refinement (placeholder for future MIAL logic)
    }
}
