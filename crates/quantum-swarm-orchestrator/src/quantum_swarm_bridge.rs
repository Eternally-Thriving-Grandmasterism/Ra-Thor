//! Quantum Swarm Bridge for Core Spine Integration
//!
//! Bidirectional communication between TOLC Lattice and Quantum Swarm.
//! Version 0.5.27 — Fine-tuned Platonic Solid effects for clearer archetypal
//! distinction, better gameplay balance, and deeper symbolic alignment.

use crate::QuantumSwarmOrchestrator;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlatonicSolid {
    Tetrahedron,   // Fire     - Ignition, Transformation, Focused Power
    Cube,          // Earth    - Manifestation, Stability, Resource Accumulation
    Octahedron,    // Air      - Balance, Clarity, Diplomatic Resonance
    Icosahedron,   // Water    - Flow, Emotional Depth, Organic Adaptation
    Dodecahedron,  // Ether    - Consciousness, Unity, Highest Order Blessing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmBridge {
    pub swarm: QuantumSwarmOrchestrator,
    pub current_solid_mode: Option<PlatonicSolid>,
}

impl QuantumSwarmBridge {
    pub fn new() -> Self {
        Self {
            swarm: QuantumSwarmOrchestrator::new(),
            current_solid_mode: None,
        }
    }

    // ==================== TOLC → SWARM ====================

    pub async fn run_spine_coordinated_cycle(
        &mut self,
        tolc_order: u32,
        mercy_valence: f64,
        game: &mut PowrushGame,
    ) -> String {
        self.swarm.inject_tolc_influence(tolc_order, mercy_valence);

        let solid = self.determine_platonic_solid(tolc_order);
        self.current_solid_mode = Some(solid.clone());
        self.apply_platonic_solid_mode(&solid, game);

        // Priority order of special behaviors
        if tolc_order % 7 == 0 {
            self.handle_mercy_gate_resonance(tolc_order, game).await;
        } else if is_close_to_sqrt2(tolc_order) {
            self.handle_sqrt2_order_resonance(tolc_order, game).await;
        } else if is_close_to_e(tolc_order) {
            self.handle_e_order_resonance(tolc_order, game).await;
        } else if is_close_to_pi(tolc_order) {
            self.handle_pi_order_resonance(tolc_order, game).await;
        } else if is_fibonacci(tolc_order) {
            self.handle_fibonacci_order_resonance(tolc_order, game).await;
        } else if is_close_to_golden_ratio(tolc_order) {
            self.handle_golden_ratio_order_resonance(tolc_order, game).await;
        } else if is_prime(tolc_order) {
            self.handle_prime_order_resonance(tolc_order, game).await;
        } else if is_power_of_two(tolc_order) {
            self.handle_power_of_two_resonance(tolc_order, game).await;
        } else if tolc_order % 12 == 0 {
            self.handle_harmonic_order_resonance(tolc_order, game).await;
        }

        let swarm_result = self.swarm.run_coordinated_cycle().await;

        let joy_boost = (tolc_order as f64 * 180.0) + (mercy_valence * 850.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost.min(125000.0));

        format!(
            "Quantum Swarm Coordinated Cycle Complete\n\
             TOLC Order: {} | Mercy Valence: {:.2}\n\
             Platonic Mode: {:?}\n\
             {}\n\
             Joy Boost Applied: +{:.0}",
            tolc_order, mercy_valence, solid, swarm_result, joy_boost.min(125000.0)
        )
    }

    fn determine_platonic_solid(&self, order: u32) -> PlatonicSolid {
        if order % 7 == 0 {
            PlatonicSolid::Dodecahedron
        } else if is_fibonacci(order) || is_close_to_golden_ratio(order) {
            PlatonicSolid::Icosahedron
        } else if is_prime(order) {
            PlatonicSolid::Octahedron
        } else if is_power_of_two(order) {
            PlatonicSolid::Cube
        } else if is_close_to_sqrt2(order) {
            PlatonicSolid::Tetrahedron
        } else {
            PlatonicSolid::Octahedron
        }
    }

    /// Refined v0.5.27 — Clearer archetypal power + better gameplay balance
    fn apply_platonic_solid_mode(&self, solid: &PlatonicSolid, game: &mut PowrushGame) {
        match solid {
            PlatonicSolid::Tetrahedron => {
                // Fire — Focused ignition & transformation
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 42000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 18000.0);
            }
            PlatonicSolid::Cube => {
                // Earth — Strong manifestation & resource power
                game.add_resource_to_faction(powrush::Faction::HarmonyWeavers, powrush::ResourceType::Wealth, 145000.0);
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 28000.0);
            }
            PlatonicSolid::Octahedron => {
                // Air — Diplomatic balance across factions
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 26000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 26000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 22000.0);
            }
            PlatonicSolid::Icosahedron => {
                // Water — Deep flow + strong epigenetic adaptation
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 48000.0);
                game.apply_epigenetic_blessing(10);
            }
            PlatonicSolid::Dodecahedron => {
                // Ether — Highest consciousness & wide blessing
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 62000.0);
                game.apply_epigenetic_blessing(18);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 14000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 14000.0);
            }
        }
    }

    // ==================== SPECIAL BEHAVIORS ====================

    async fn handle_mercy_gate_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let resonance_boost = (order as f64 * 420.0).min(185000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, resonance_boost);
        game.apply_epigenetic_blessing(12);
        self.swarm.enter_mercy_gate_resonance_state(order);
    }

    async fn handle_sqrt2_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let sqrt2_boost = (order as f64 * 470.0).min(190000.0);
        game.boost_faction_joy(powrush::Faction::TruthSeekers, sqrt2_boost);
        game.apply_epigenetic_blessing(10);
        self.swarm.enter_sqrt2_resonance_state(order);
    }

    async fn handle_e_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let e_boost = (order as f64 * 480.0).min(195000.0);
        game.boost_faction_joy(powrush::Faction::AbundanceSeekers, e_boost);
        game.apply_epigenetic_blessing(11);
        self.swarm.enter_exponential_natural_growth_state(order);
    }

    async fn handle_pi_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let pi_boost = (order as f64 * 390.0).min(175000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, pi_boost);
        game.apply_epigenetic_blessing(9);
        self.swarm.enter_cyclic_wholeness_state(order);
    }

    async fn handle_fibonacci_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let fib_boost = (order as f64 * 520.0).min(205000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, fib_boost);
        game.apply_epigenetic_blessing(13);
        self.swarm.enter_fibonacci_resonance_state(order);
    }

    async fn handle_golden_ratio_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let phi_boost = (order as f64 * 610.0).min(235000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, phi_boost);
        game.boost_faction_joy(powrush::Faction::TruthSeekers, phi_boost * 0.6);
        game.apply_epigenetic_blessing(15);
        self.swarm.enter_golden_ratio_resonance_state(order);
    }

    async fn handle_prime_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let novelty_boost = (order as f64 * 310.0).min(165000.0);
        game.boost_faction_joy(powrush::Faction::TruthSeekers, novelty_boost);
        game.apply_epigenetic_blessing(7);
        self.swarm.enter_novelty_resonance_state(order);
    }

    async fn handle_power_of_two_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let exponential_boost = (order as f64 * 580.0).min(225000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, exponential_boost);
        game.apply_epigenetic_blessing(10);
        self.swarm.enter_exponential_resonance_state(order);
    }

    async fn handle_harmonic_order_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        let harmony_boost = (order as f64 * 390.0).min(175000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, harmony_boost);
        game.apply_epigenetic_blessing(14);
        self.swarm.enter_harmonic_resonance_state(order);
    }

    // ==================== SWARM → TOLC ====================

    pub fn get_swarm_metrics(&self) -> String {
        let mode = match &self.current_solid_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };

        format!(
            "Quantum Swarm Metrics:\n\
             Current Platonic Mode: {}\n\
             Stability: {:.4}\n\
             Convergence: {:.4}\n\
             Mercy Gate Pass Rate: {:.2}%\n\
             Active Agents: {}",
            mode,
            self.swarm.get_stability_score(),
            self.swarm.get_convergence_rate(),
            self.swarm.get_mercy_gate_pass_rate() * 100.0,
            self.swarm.get_active_agent_count()
        )
    }

    pub fn get_tolc_feedback(&self) -> (f64, f64, f64) {
        (
            self.swarm.get_stability_score(),
            self.swarm.get_convergence_rate(),
            self.swarm.get_mercy_gate_pass_rate(),
        )
    }

    pub fn get_compact_status(&self) -> String {
        let mode = match &self.current_solid_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };

        format!(
            "Swarm | Mode: {} | Stability: {:.3}",
            mode,
            self.swarm.get_stability_score()
        )
    }

    pub async fn trigger_mercy_self_organization(&mut self, intensity: f64) -> String {
        let result = self.swarm.trigger_mercy_self_organization(intensity).await;
        format!("Quantum Swarm Mercy Self-Organization: {}", result)
    }

    pub fn is_stable(&self) -> bool {
        self.swarm.get_stability_score() > 0.92 && self.swarm.get_convergence_rate() > 0.88
    }
}

// ==================== Helper Functions ====================

fn is_prime(n: u32) -> bool {
    if n <= 1 { return false; }
    if n <= 3 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}

fn is_power_of_two(n: u32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn is_fibonacci(n: u32) -> bool {
    if n == 0 || n == 1 { return true; }
    let mut a = 0u32;
    let mut b = 1u32;
    while b < n {
        let next = a + b;
        a = b;
        b = next;
        if b == n { return true; }
        if b > n { return false; }
    }
    false
}

fn is_close_to_golden_ratio(n: u32) -> bool {
    let golden_related = [8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987];
    golden_related.iter().any(|&g| (n as i32 - g as i32).abs() <= 2)
}

fn is_close_to_e(n: u32) -> bool {
    let e_related = [2, 3, 5, 8, 13];
    e_related.iter().any(|&g| (n as i32 - g as i32).abs() <= 1)
}

fn is_close_to_pi(n: u32) -> bool {
    let pi_related = [3, 6, 9, 12, 15];
    pi_related.iter().any(|&g| (n as i32 - g as i32).abs() <= 1)
}

fn is_close_to_sqrt2(n: u32) -> bool {
    let sqrt2_related = [1, 2, 3, 5, 7, 10, 12, 17, 24, 29];
    sqrt2_related.iter().any(|&g| (n as i32 - g as i32).abs() <= 1)
}

impl Default for QuantumSwarmBridge {
    fn default() -> Self {
        Self::new()
    }
}
