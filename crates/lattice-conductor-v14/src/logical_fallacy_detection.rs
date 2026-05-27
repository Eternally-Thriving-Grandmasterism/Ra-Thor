// crates/lattice-conductor-v14/src/logical_fallacy_detection.rs
// Production-grade Logical Fallacy Detection Module for PATSAGi
//
// Includes structural detection in ArgumentGraph + flagging system for debate.

use crate::argumentation::{ArgumentGraph, ArgumentId};
use std::collections::{HashMap, HashSet};

/// Types of logical fallacies we can detect
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FallacyType {
    UnsupportedClaim,
    CircularSupport,
    WeakEvidenceChain,
    PotentialStrawMan,
    OverrelianceOnAuthority,
}

/// Represents a detected fallacy
#[derive(Debug, Clone)]
pub struct DetectedFallacy {
    pub fallacy_type: FallacyType,
    pub target_claim_id: ArgumentId,
    pub description: String,
    pub severity: f64, // 0.0 - 1.0
}

/// Production-grade Logical Fallacy Detector
pub struct LogicalFallacyDetector;

impl LogicalFallacyDetector {
    /// Detect structural fallacies inside an ArgumentGraph
    pub fn detect_structural_fallacies(graph: &ArgumentGraph) -> Vec<DetectedFallacy> {
        let mut fallacies = Vec::new();

        // 1. Unsupported Claims (claims with no support)
        for (id, claim) in &graph.claims {
            let has_support = graph.supports.iter().any(|s| s.target_claim_id == *id);
            if !has_support {
                fallacies.push(DetectedFallacy {
                    fallacy_type: FallacyType::UnsupportedClaim,
                    target_claim_id: *id,
                    description: format!("Claim '{}' has no supporting arguments", claim.content),
                    severity: 0.6,
                });
            }
        }

        // 2. Circular Support Detection
        let circular = Self::find_circular_support(graph);
        for claim_id in circular {
            if let Some(claim) = graph.claims.get(&claim_id) {
                fallacies.push(DetectedFallacy {
                    fallacy_type: FallacyType::CircularSupport,
                    target_claim_id: claim_id,
                    description: format!("Circular reasoning detected involving claim '{}'", claim.content),
                    severity: 0.85,
                });
            }
        }

        // 3. Weak Evidence Chain (many attacks, few supports)
        for (id, claim) in &graph.claims {
            let support_count = graph.supports.iter().filter(|s| s.target_claim_id == *id).count();
            let attack_count = graph.attacks.iter().filter(|a| a.target_claim_id == *id).count();

            if attack_count > support_count + 1 {
                fallacies.push(DetectedFallacy {
                    fallacy_type: FallacyType::WeakEvidenceChain,
                    target_claim_id: *id,
                    description: format!("Claim '{}' has significantly more attacks than supports", claim.content),
                    severity: 0.7,
                });
            }
        }

        fallacies
    }

    /// Simple circular support detection using DFS
    fn find_circular_support(graph: &ArgumentGraph) -> HashSet<ArgumentId> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut circular = HashSet::new();

        for &claim_id in graph.claims.keys() {
            if !visited.contains(&claim_id) {
                Self::dfs_circular(claim_id, graph, &mut visited, &mut rec_stack, &mut circular);
            }
        }

        circular
    }

    fn dfs_circular(
        claim_id: ArgumentId,
        graph: &ArgumentGraph,
        visited: &mut HashSet<ArgumentId>,
        rec_stack: &mut HashSet<ArgumentId>,
        circular: &mut HashSet<ArgumentId>,
    ) {
        visited.insert(claim_id);
        rec_stack.insert(claim_id);

        for support in &graph.supports {
            if support.target_claim_id == claim_id {
                let supporter_id = support.id; // simplistic mapping
                if !visited.contains(&supporter_id) {
                    Self::dfs_circular(supporter_id, graph, visited, rec_stack, circular);
                } else if rec_stack.contains(&supporter_id) {
                    circular.insert(claim_id);
                }
            }
        }

        rec_stack.remove(&claim_id);
    }
}
