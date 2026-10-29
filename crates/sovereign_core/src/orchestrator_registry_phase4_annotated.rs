//! Phase 4: Annotated Logic Lattice + Enhanced Contradiction Resolution
//!
//! Extends the ParaconsistentSuperKernel with PAL2v-style 4-valued annotated logic
//! for finer-grained, non-explosive contradiction handling.

use std::collections::HashMap;

// Re-use previous structs from Phase 3 (abbreviated for clarity)
// In full implementation, include all previous code here.

#[derive(Debug, Clone, PartialEq)]
pub enum AnnotatedState {
    True,           // t – high confidence, no contradiction
    False,          // f – low confidence, no contradiction
    Inconsistent,   // ⊤ – both true and false (high severity)
    Indeterminate,  // ⊥ – insufficient data
}

impl AnnotatedState {
    pub fn from_valence(valence: f64) -> Self {
        if valence >= 0.999999 {
            AnnotatedState::True
        } else if valence < 0.5 {
            AnnotatedState::False
        } else if valence < 0.999 {
            AnnotatedState::Inconsistent
        } else {
            AnnotatedState::Indeterminate
        }
    }

    pub fn is_contradictory(&self) -> bool {
        matches!(self, AnnotatedState::Inconsistent)
    }
}

// Extend ParaconsistentSuperKernel with Phase 4 method
impl ParaconsistentSuperKernel {
    pub fn consume_feed_with_annotated_logic(&mut self, feed: &ParaconsistentFeed) -> Vec<ParaconsistentAction> {
        let mut actions = self.consume_feed(feed); // Base Phase 3 logic

        // Phase 4: Annotated logic enhancement
        for report in &feed.high_severity_contradictions {
            let state = AnnotatedState::from_valence(report.severity);
            if state.is_contradictory() {
                actions.push(ParaconsistentAction::ResolveContradiction {
                    report: report.clone(),
                    resolution: "Apply annotated lattice resolution: promote to higher-order harmony via 7 Mercy Gates".to_string(),
                });
            }
        }

        // Additional annotated-based abundance check
        if feed.abundance_ready && feed.global_maat > 0.95 {
            actions.push(ParaconsistentAction::TriggerAbundanceDistribution {
                reason: "Annotated lattice confirms high Ma’at + Lumenas CI".to_string(),
                intensity: feed.global_maat,
            });
        }

        actions
    }
}

// Extend SovereignCore
impl SovereignCore {
    pub fn run_paraconsistent_cycle_annotated(&mut self) -> Vec<ParaconsistentAction> {
        let feed = self.registry.get_paraconsistent_feed();
        self.super_kernel.consume_feed_with_annotated_logic(&feed)
    }
}