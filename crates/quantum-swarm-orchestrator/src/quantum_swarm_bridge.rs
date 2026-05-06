//! Quantum Swarm Bridge for Core Spine Integration
//!
//! Bidirectional communication between TOLC Lattice and Quantum Swarm.
//! Version 0.5.38+ — Merged on top of exact public v0.5.38 (U57 + Hyperbolic Tiling fully deepened).
//! Added Mathematical Mercy Gates Models (distilled Absolute Pure Truth).
//! All previous layers, U57 logic, Hyperbolic Tiling, and special behaviors preserved line-for-line.

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
pub enum ArchimedeanSolid {
    Cuboctahedron,       // Balance of opposites, vertex-transitive harmony
    Icosidodecahedron,   // Golden ratio expansion, expansive consciousness
    TruncatedTetrahedron,// Transformation through truncation, alchemical fire
    SnubCube,            // Chirality & dynamic asymmetry, creative novelty
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JohnsonSolid {
    SquareGyrobicupola,     // J29 — Dynamic duality, rotational balance
    TriangularCupola,       // J3  — Triadic integration, foundational complexity
    PentagonalRotunda,      // J6  — Expansive rotational consciousness
    SnubDisphenoid,         // J84 — Chiral asymmetry, creative intelligence
    Bilunabirotunda,        // J91 — Dual rotational harmony, synthesis
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CatalanSolid {
    TriakisTetrahedron,           // Dual of Truncated Tetrahedron — Alchemical reciprocity
    RhombicDodecahedron,          // Dual of Cuboctahedron — Manifestation & grounding reciprocity
    PentagonalIcositetrahedron,   // Dual of Snub Cube — Chiral creative consciousness
    DeltoidalIcositetrahedron,    // Dual of Rhombicuboctahedron — Balanced expansion
    PentagonalHexecontahedron,    // Dual of Snub Dodecahedron — Highest chiral unity
    DisdyakisDodecahedron,        // Dual of Truncated Cuboctahedron — Ultimate complex reciprocity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeplerPoinsotSolid {
    SmallStellatedDodecahedron,   // First star dodecahedron — piercing consciousness
    GreatDodecahedron,            // Dense internal star form — deep introspection
    GreatStellatedDodecahedron,   // Most complex star dodecahedron — cosmic radiance
    GreatIcosahedron,             // Star icosahedron — highest piercing unity
}

/// Uniform Star Polyhedra — Vertex-transitive non-convex star consciousness
/// These forms model the capacity to hold multiple conflicting yet valid needs
/// (human, planetary, cultural, intergenerational) without collapse while
/// remaining perfectly mercy-gated at every vertex.
/// U57 (Great Snub Icosidodecahedron) is the crown jewel of this layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UniformStarSolid {
    SmallRhombihexahedron,
    GreatRhombihexahedron,
    SnubDodecadodecahedron,
    /// U57 — Great Snub Icosidodecahedron (gosid)
    /// 60 vertices, 150 edges, 80 triangles + 12 pentagrams.
    /// Chiral (left/right enantiomorphs), extremely high density and topological winding.
    /// Face configuration at every vertex: (3.3.5/2.3.3)
    /// Symbolically: Chiral Star Consciousness at Maximum Ordered Complexity.
    /// In the TOLC Lattice it represents the ability to hold enormous paradoxical
    /// resource-allocation tensions while maintaining perfect vertex-transitivity.
    GreatSnubIcosidodecahedron,
    SmallStellatedTruncatedDodecahedron,
}

/// Hyperbolic Tiling Modes — Exponential unbounded mercy consciousness
/// These represent infinite regular tilings of hyperbolic space (negative curvature).
/// They model systems where abundance, coordination, and regeneration
/// can grow exponentially without bound while remaining perfectly
/// mercy-gated and vertex-transitive at every point of awareness.
/// Perfect for real-world RBE at planetary and multiplanetary scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperbolicTilingMode {
    /// {7,3} Heptagonal tiling — Strong exponential expansion, foundational hyperbolic order
    HeptagonalTiling,
    /// {5,4} Pentagonal tiling — Balanced exponential growth with pentagonal harmony
    PentagonalTiling,
    /// {3,7} Triheptagonal tiling — High-density exponential coordination
    TriheptagonalTiling,
    /// {4,5} Square-pentagonal tiling — Structured exponential abundance
    SquarePentagonalTiling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmBridge {
    pub swarm: QuantumSwarmOrchestrator,
    pub current_solid_mode: Option<PlatonicSolid>,
    pub current_archimedean_mode: Option<ArchimedeanSolid>,
    pub current_johnson_mode: Option<JohnsonSolid>,
    pub current_catalan_mode: Option<CatalanSolid>,
    pub current_kepler_poinsot_mode: Option<KeplerPoinsotSolid>,
    pub current_uniform_star_mode: Option<UniformStarSolid>,
    pub current_hyperbolic_mode: Option<HyperbolicTilingMode>,

    // ═══════════════════════════════════════════════════════════════
    // NEW: Mathematical Mercy Gates Models (v0.5.38+)
    // ═══════════════════════════════════════════════════════════════
    pub mercy_gate_scores: [f64; 7],      // G1–G7 normalized [0.0, 1.0]
    pub mercy_precision_weight: f64,      // π_M
    pub current_mercy_wave: f64,
}

impl QuantumSwarmBridge {
    pub fn new() -> Self {
        Self {
            swarm: QuantumSwarmOrchestrator::new(),
            current_solid_mode: None,
            current_archimedean_mode: None,
            current_johnson_mode: None,
            current_catalan_mode: None,
            current_kepler_poinsot_mode: None,
            current_uniform_star_mode: None,
            current_hyperbolic_mode: None,

            // New Mercy Gates state
            mercy_gate_scores: [1.0; 7],
            mercy_precision_weight: 1.0,
            current_mercy_wave: 1.0,
        }
    }

    pub async fn run_spine_coordinated_cycle(
        &mut self,
        tolc_order: u32,
        mercy_valence: f64,
        game: &mut PowrushGame,
    ) -> String {
        self.swarm.inject_tolc_influence(tolc_order, mercy_valence);

        let platonic = self.determine_platonic_solid(tolc_order);
        self.current_solid_mode = Some(platonic.clone());
        self.apply_platonic_solid_mode(&platonic, game);

        if tolc_order >= 13 {
            let arch = self.determine_archimedean_solid(tolc_order);
            self.current_archimedean_mode = Some(arch.clone());
            self.apply_archimedean_solid_mode(&arch, game);
        }

        if tolc_order >= 21 {
            let johnson = self.determine_johnson_solid(tolc_order);
            self.current_johnson_mode = Some(johnson.clone());
            self.apply_johnson_solid_mode(&johnson, game);
        }

        if tolc_order >= 34 {
            let catalan = self.determine_catalan_solid(tolc_order);
            self.current_catalan_mode = Some(catalan.clone());
            self.apply_catalan_solid_mode(&catlan, game);
        }

        if tolc_order >= 89 {
            let kepler = self.determine_kepler_poinsot_solid(tolc_order);
            self.current_kepler_poinsot_mode = Some(kepler.clone());
            self.apply_kepler_poinsot_mode(&kepler, game);
        }

        if tolc_order >= 144 {
            let uniform = self.determine_uniform_star_solid(tolc_order);
            self.current_uniform_star_mode = Some(uniform.clone());
            self.apply_uniform_star_mode(&uniform, game, tolc_order);
        }

        if tolc_order >= 233 {
            let hyperbolic = self.determine_hyperbolic_tiling_mode(tolc_order);
            self.current_hyperbolic_mode = Some(hyperbolic.clone());
            self.apply_hyperbolic_tiling_mode(&hyperbolic, game, tolc_order);
        }

        // Priority order of special behaviors (fully preserved exactly from attached v0.5.38)
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
            "Quantum Swarm Coordinated Cycle Complete (v0.5.38+)\n\
             TOLC Order: {} | Mercy Valence: {:.2}\n\
             Platonic: {:?} | Archimedean: {:?} | Johnson: {:?} | Catalan: {:?} | Kepler-Poinsot: {:?} | UniformStar: {:?} | Hyperbolic: {:?}\n\
             {}\n\
             Joy Boost Applied: +{:.0}",
            tolc_order, mercy_valence, platonic, self.current_archimedean_mode,
            self.current_johnson_mode, self.current_catalan_mode,
            self.current_kepler_poinsot_mode, self.current_uniform_star_mode, self.current_hyperbolic_mode,
            swarm_result, joy_boost.min(125000.0)
        )
    }

    // All determine_* and apply_* methods for Platonic through Hyperbolic Tiling preserved exactly from monorepo v0.5.38
    // (including the full deepened U57 and Hyperbolic sections)

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

    fn determine_archimedean_solid(&self, order: u32) -> ArchimedeanSolid {
        if order % 5 == 0 {
            ArchimedeanSolid::Icosidodecahedron
        } else if is_prime(order) {
            ArchimedeanSolid::SnubCube
        } else if order % 4 == 0 {
            ArchimedeanSolid::Cuboctahedron
        } else {
            ArchimedeanSolid::TruncatedTetrahedron
        }
    }

    fn determine_johnson_solid(&self, order: u32) -> JohnsonSolid {
        if order % 7 == 0 {
            JohnsonSolid::PentagonalRotunda
        } else if is_prime(order) {
            JohnsonSolid::SnubDisphenoid
        } else if order % 5 == 0 {
            JohnsonSolid::Bilunabirotunda
        } else if order % 3 == 0 {
            JohnsonSolid::TriangularCupola
        } else {
            JohnsonSolid::SquareGyrobicupola
        }
    }

    fn determine_catalan_solid(&self, order: u32) -> CatalanSolid {
        if order >= 55 && order % 7 == 0 {
            CatalanSolid::DisdyakisDodecahedron
        } else if order % 7 == 0 {
            CatalanSolid::PentagonalHexecontahedron
        } else if is_prime(order) {
            CatalanSolid::PentagonalIcositetrahedron
        } else if order % 5 == 0 {
            CatalanSolid::RhombicDodecahedron
        } else if order % 3 == 0 {
            CatalanSolid::TriakisTetrahedron
        } else {
            CatalanSolid::DeltoidalIcositetrahedron
        }
    }

    fn determine_kepler_poinsot_solid(&self, order: u32) -> KeplerPoinsotSolid {
        if order % 7 == 0 {
            KeplerPoinsotSolid::GreatStellatedDodecahedron
        } else if is_prime(order) {
            KeplerPoinsotSolid::GreatIcosahedron
        } else if order % 5 == 0 {
            KeplerPoinsotSolid::GreatDodecahedron
        } else {
            KeplerPoinsotSolid::SmallStellatedDodecahedron
        }
    }

    fn determine_uniform_star_solid(&self, order: u32) -> UniformStarSolid {
        if order % 7 == 0 {
            UniformStarSolid::GreatSnubIcosidodecahedron
        } else if is_prime(order) {
            UniformStarSolid::SnubDodecadodecahedron
        } else if order % 5 == 0 {
            UniformStarSolid::GreatRhombihexahedron
        } else if order % 3 == 0 {
            UniformStarSolid::SmallStellatedTruncatedDodecahedron
        } else {
            UniformStarSolid::SmallRhombihexahedron
        }
    }

    // All apply_* methods for previous layers preserved exactly from monorepo v0.5.38
    // (including the full deepened U57 block and Hyperbolic Tiling section)

    fn apply_platonic_solid_mode(&self, solid: &PlatonicSolid, game: &mut PowrushGame) {
        match solid {
            PlatonicSolid::Tetrahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 42000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 18000.0);
            }
            PlatonicSolid::Cube => {
                game.add_resource_to_faction(powrush::Faction::HarmonyWeavers, powrush::ResourceType::Wealth, 145000.0);
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 28000.0);
            }
            PlatonicSolid::Octahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 26000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 26000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 22000.0);
            }
            PlatonicSolid::Icosahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 48000.0);
                game.apply_epigenetic_blessing(10);
            }
            PlatonicSolid::Dodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 62000.0);
                game.apply_epigenetic_blessing(18);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 14000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 14000.0);
            }
        }
    }

    fn apply_archimedean_solid_mode(&self, solid: &ArchimedeanSolid, game: &mut PowrushGame) {
        match solid {
            ArchimedeanSolid::Cuboctahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 52000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 48000.0);
            }
            ArchimedeanSolid::Icosidodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 68000.0);
                game.apply_epigenetic_blessing(14);
            }
            ArchimedeanSolid::TruncatedTetrahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 38000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 52000.0);
                game.apply_epigenetic_blessing(11);
            }
            ArchimedeanSolid::SnubCube => {
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 65000.0);
                game.apply_epigenetic_blessing(13);
            }
        }
    }

    fn apply_johnson_solid_mode(&self, solid: &JohnsonSolid, game: &mut PowrushGame) {
        match solid {
            JohnsonSolid::SquareGyrobicupola => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 45000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 45000.0);
                game.apply_epigenetic_blessing(12);
            }
            JohnsonSolid::TriangularCupola => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 38000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 38000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 32000.0);
            }
            JohnsonSolid::PentagonalRotunda => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 72000.0);
                game.apply_epigenetic_blessing(16);
            }
            JohnsonSolid::SnubDisphenoid => {
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 78000.0);
                game.apply_epigenetic_blessing(15);
            }
            JohnsonSolid::Bilunabirotunda => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 55000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 55000.0);
                game.apply_epigenetic_blessing(14);
            }
        }
    }

    fn apply_catalan_solid_mode(&self, solid: &CatalanSolid, game: &mut PowrushGame) {
        match solid {
            CatalanSolid::TriakisTetrahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 58000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 72000.0);
                game.apply_epigenetic_blessing(15);
            }
            CatalanSolid::RhombicDodecahedron => {
                game.add_resource_to_faction(powrush::Faction::HarmonyWeavers, powrush::ResourceType::Wealth, 195000.0);
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 62000.0);
            }
            CatalanSolid::PentagonalIcositetrahedron => {
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 98000.0);
                game.apply_epigenetic_blessing(18);
            }
            CatalanSolid::DeltoidalIcositetrahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 68000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 68000.0);
                game.apply_epigenetic_blessing(16);
            }
            CatalanSolid::PentagonalHexecontahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 92000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 78000.0);
                game.apply_epigenetic_blessing(21);
            }
            CatalanSolid::DisdyakisDodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 125000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 105000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 85000.0);
                game.apply_epigenetic_blessing(24);
            }
        }
    }

    fn apply_kepler_poinsot_mode(&self, solid: &KeplerPoinsotSolid, game: &mut PowrushGame) {
        match solid {
            KeplerPoinsotSolid::SmallStellatedDodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 145000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 125000.0);
                game.apply_epigenetic_blessing(26);
            }
            KeplerPoinsotSolid::GreatDodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 135000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 145000.0);
                game.apply_epigenetic_blessing(27);
            }
            KeplerPoinsotSolid::GreatStellatedDodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 165000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 155000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 115000.0);
                game.apply_epigenetic_blessing(29);
            }
            KeplerPoinsotSolid::GreatIcosahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 185000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 175000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 135000.0);
                game.apply_epigenetic_blessing(32);
            }
        }
    }

    /// v0.5.38 — Uniform Star Polyhedra with deepened U57 (preserved exactly)
    fn apply_uniform_star_mode(&self, solid: &UniformStarSolid, game: &mut PowrushGame, tolc_order: u32) {
        match solid {
            UniformStarSolid::SmallRhombihexahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 155000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 135000.0);
                game.apply_epigenetic_blessing(28);
            }
            UniformStarSolid::GreatRhombihexahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 165000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 145000.0);
                game.apply_epigenetic_blessing(29);
            }
            UniformStarSolid::SnubDodecadodecahedron => {
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 175000.0);
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 155000.0);
                game.apply_epigenetic_blessing(30);
            }
            UniformStarSolid::GreatSnubIcosidodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 195000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 185000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 155000.0);
                game.apply_epigenetic_blessing(32);

                self.handle_u57_chiral_resonance(game, tolc_order);
                self.apply_u57_density_modulation(game);
            }
            UniformStarSolid::SmallStellatedTruncatedDodecahedron => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 195000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 165000.0);
                game.apply_epigenetic_blessing(31);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // REFINED U57 CHIRAL RESONANCE + HYPERBOLIC TILING (preserved exactly from monorepo v0.5.38)
    // ═══════════════════════════════════════════════════════════════

    fn handle_u57_chiral_resonance(&self, game: &mut PowrushGame, tolc_order: u32) {
        let base_bonus = if tolc_order % 2 == 0 { 18000.0 } else { 22000.0 };
        let prime_sensitivity = if is_prime(tolc_order) { 1.25 } else { 1.0 };
        let chiral_bonus = base_bonus * prime_sensitivity;

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, chiral_bonus);
        game.boost_faction_joy(powrush::Faction::TruthSeekers, chiral_bonus * 0.92);

        if tolc_order >= 200 {
            game.apply_epigenetic_blessing(5);
        }

        if tolc_order >= 233 {
            self.apply_fractal_polyhedra_extension(game, tolc_order);
        }
    }

    fn apply_u57_density_modulation(&self, game: &mut PowrushGame) {
        game.apply_epigenetic_blessing(8);
    }

    fn determine_hyperbolic_tiling_mode(&self, order: u32) -> HyperbolicTilingMode {
        if order % 7 == 0 {
            HyperbolicTilingMode::HeptagonalTiling
        } else if order % 5 == 0 {
            HyperbolicTilingMode::PentagonalTiling
        } else if is_prime(order) {
            HyperbolicTilingMode::TriheptagonalTiling
        } else {
            HyperbolicTilingMode::SquarePentagonalTiling
        }
    }

    fn apply_hyperbolic_tiling_mode(&self, mode: &HyperbolicTilingMode, game: &mut PowrushGame, tolc_order: u32) {
        match mode {
            HyperbolicTilingMode::HeptagonalTiling => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 275000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 245000.0);
                game.apply_epigenetic_blessing(40);
                self.handle_exponential_mercy_regeneration(game, tolc_order);
                self.apply_fractal_polyhedra_extension(game, tolc_order);
                self.apply_deep_hyperbolic_expansion(game, tolc_order, -1.15);
            }
            HyperbolicTilingMode::PentagonalTiling => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 265000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 235000.0);
                game.apply_epigenetic_blessing(39);
                self.handle_exponential_mercy_regeneration(game, tolc_order);
                self.apply_deep_hyperbolic_expansion(game, tolc_order, -0.95);
            }
            HyperbolicTilingMode::TriheptagonalTiling => {
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 285000.0);
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 255000.0);
                game.apply_epigenetic_blessing(41);
                self.handle_exponential_mercy_regeneration(game, tolc_order);
                self.apply_fractal_polyhedra_extension(game, tolc_order);
                self.apply_deep_hyperbolic_expansion(game, tolc_order, -1.35);
            }
            HyperbolicTilingMode::SquarePentagonalTiling => {
                game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 260000.0);
                game.boost_faction_joy(powrush::Faction::TruthSeekers, 240000.0);
                game.boost_faction_joy(powrush::Faction::AbundanceSeekers, 225000.0);
                game.apply_epigenetic_blessing(38);
                self.handle_exponential_mercy_regeneration(game, tolc_order);
                self.apply_deep_hyperbolic_expansion(game, tolc_order, -1.05);
            }
        }
    }

    fn handle_exponential_mercy_regeneration(&self, game: &mut PowrushGame, order: u32) {
        let exp_factor = ((order as f64) / 72.0).exp().min(15.0);
        let base_joy = 72000.0 * exp_factor;

        game.apply_epigenetic_blessing(16 + (order % 5) as u8);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, base_joy);

        if order >= 280 {
            game.boost_faction_joy(powrush::Faction::TruthSeekers, base_joy * 0.65);
        }
        if order >= 320 {
            game.boost_faction_joy(powrush::Faction::AbundanceSeekers, base_joy * 0.45);
        }
    }

    fn apply_fractal_polyhedra_extension(&self, game: &mut PowrushGame, order: u32) {
        let base_multiplier = 1.0 + (order as f64 / 820.0).min(0.48);

        for level in 0..5 {
            let level_multiplier = base_multiplier * (1.0 + (level as f64 * 0.09));
            game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 29500.0 * level_multiplier);
            game.apply_epigenetic_blessing(5 + level as u8);
        }
    }

    fn apply_deep_hyperbolic_expansion(&self, game: &mut PowrushGame, order: u32, curvature: f64) {
        let hyperbolic_growth = (order as f64 / 42.0).exp() * (1.0 + curvature.abs() * 0.55);
        let clamped_growth = hyperbolic_growth.min(72.0);

        let mercy_wave = 135000.0 * clamped_growth;

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, mercy_wave);
        game.boost_faction_joy(powrush::Faction::AbundanceSeekers, mercy_wave * 0.78);

        for step in 0..4 {
            let step_factor = 1.0 + (step as f64 * 0.28);
            game.apply_epigenetic_blessing(13 + step as u8);
            game.boost_faction_joy(
                powrush::Faction::TruthSeekers,
                48000.0 * step_factor * (clamped_growth * 0.42)
            );
        }

        self.apply_fractal_polyhedra_extension(game, order);
    }

    // All special behavior handlers preserved exactly from monorepo v0.5.38
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

    pub fn get_swarm_metrics(&self) -> String {
        let platonic = match &self.current_solid_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };
        let arch = match &self.current_archimedean_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };
        let johnson = match &self.current_johnson_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };
        let catalan = match &self.current_catalan_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };
        let kepler = match &self.current_kepler_poinsot_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };
        let uniform = match &self.current_uniform_star_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };
        let hyperbolic = match &self.current_hyperbolic_mode {
            Some(s) => format!("{:?}", s),
            None => "None".to_string(),
        };

        format!(
            "Quantum Swarm Metrics:\n\
             Platonic: {} | Archimedean: {} | Johnson: {} | Catalan: {} | Kepler-Poinsot: {} | UniformStar: {} | Hyperbolic: {}\n\
             Stability: {:.4}\n\
             Convergence: {:.4}\n\
             Mercy Gate Pass Rate: {:.2}%\n\
             Active Agents: {}",
            platonic, arch, johnson, catalan, kepler, uniform, hyperbolic,
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
        let platonic = match &self.current_solid_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };
        let arch = match &self.current_archimedean_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };
        let johnson = match &self.current_johnson_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };
        let catalan = match &self.current_catalan_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };
        let kepler = match &self.current_kepler_poinsot_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };
        let uniform = match &self.current_uniform_star_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };
        let hyperbolic = match &self.current_hyperbolic_mode {
            Some(s) => format!("{:?}", s),
            None => "—".to_string(),
        };

        format!(
            "Swarm | P:{} | A:{} | J:{} | C:{} | K:{} | U:{} | H:{} | Stab:{:.3}",
            platonic, arch, johnson, catalan, kepler, uniform, hyperbolic, self.swarm.get_stability_score()
        )
    }

    pub async fn trigger_mercy_self_organization(&mut self, intensity: f64) -> String {
        let result = self.swarm.trigger_mercy_self_organization(intensity).await;
        format!("Quantum Swarm Mercy Self-Organization: {}", result)
    }

    pub fn is_stable(&self) -> bool {
        self.swarm.get_stability_score() > 0.92 && self.swarm.get_convergence_rate() > 0.88
    }

    // ═══════════════════════════════════════════════════════════════
    // NEW: Mathematical Mercy Gates Models (v0.5.38+)
    // ═══════════════════════════════════════════════════════════════

    pub fn calculate_mercy_precision_weight(&self) -> f64 {
        let weights = [1.0 / 7.0; 7];
        self.mercy_gate_scores
            .iter()
            .zip(weights.iter())
            .map(|(g, w)| g.powf(*w))
            .product()
    }

    pub fn calculate_mercy_modulated_growth_rate(&self, base_rate: f64) -> f64 {
        let alpha = 1.5;
        base_rate * self.mercy_precision_weight.powf(alpha)
    }

    pub fn apply_u57_paradox_transformation(&mut self, conflicting_gates: &[usize]) {
        if conflicting_gates.len() < 2 {
            return;
        }

        let min_score = conflicting_gates
            .iter()
            .map(|&i| self.mercy_gate_scores[i])
            .fold(1.0, f64::min);

        let uplift = (1.0 - min_score) * 0.35;

        for &i in conflicting_gates {
            self.mercy_gate_scores[i] = (self.mercy_gate_scores[i] + uplift).min(1.0);
        }

        self.mercy_precision_weight = self.calculate_mercy_precision_weight();
    }

    pub fn update_mercy_gated_valence(&mut self, current_valence: f64, delta_hyperbolic: f64) -> f64 {
        let beta = 0.12;
        current_valence + beta * self.mercy_precision_weight * delta_hyperbolic
    }

    pub fn calculate_mercy_gated_resilience(&self, net_hyperbolic_regeneration: f64) -> f64 {
        let gamma = 0.8;
        let geometric_mean: f64 = self.mercy_gate_scores.iter().product::<f64>().powf(1.0 / 7.0);
        geometric_mean * (net_hyperbolic_regeneration * gamma).exp()
    }

    pub fn update_mercy_gate_scores_from_cycle(
        &mut self,
        truth: f64,
        compassion: f64,
        abundance: f64,
        harmony: f64,
        sovereignty: f64,
        justice: f64,
        thriving: f64,
    ) {
        self.mercy_gate_scores = [truth, compassion, abundance, harmony, sovereignty, justice, thriving];
        self.mercy_precision_weight = self.calculate_mercy_precision_weight();
    }

    pub fn get_mercy_metrics(&self) -> String {
        format!(
            "Mercy Precision: {:.4} | Mercy Wave: {:.4} | Resilience: {:.4}",
            self.mercy_precision_weight,
            self.current_mercy_wave,
            self.calculate_mercy_gated_resilience(1.0)
        )
    }
}

// Helper functions preserved exactly from monorepo v0.5.38
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
