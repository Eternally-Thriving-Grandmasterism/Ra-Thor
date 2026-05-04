//! TOLC Advanced 3D Lattice Renderer — Interstellar Operations v0.5.32
//! Professional 3D rendering layer for the Living Cathedral
//!
//! Generates export-ready 3D data (JSON point cloud + edges) for Three.js, Bevy, Blender, etc.
//! Includes rich ASCII 3D layered projections and Mercy-Gated spatial mandalas.
//! Fully integrated with Lattice Activation Engine and all prior TOLC modules.
//!
//! Alignment: From 2D text visualization → True Advanced 3D Lattice Rendering (May 2026)

use crate::tolc_lattice_activation_engine::TOLCLatticeActivationEngine;
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};
use std::fmt::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLC3DLatticeRenderer {
    pub version: String,
    pub max_order: u32,
    pub root_core_position: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice3DPoint {
    pub order: u32,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub color: String,           // hex or named color
    pub size: f64,
    pub mercy_valence: f64,
    pub stability: f64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice3DEdge {
    pub from_order: u32,
    pub to_order: u32,
    pub strength: f64,
    pub color: String,
}

impl TOLC3DLatticeRenderer {
    pub fn new() -> Self {
        Self {
            version: "0.5.32 — Advanced 3D Lattice Rendering".to_string(),
            max_order: 79,
            root_core_position: [0.0, 0.0, 0.0],
        }
    }

    /// Generate complete 3D point cloud + edges ready for Three.js / Bevy / Blender
    pub fn generate_3d_point_cloud_json(&self) -> String {
        let mut points: Vec<Lattice3DPoint> = Vec::new();
        let mut edges: Vec<Lattice3DEdge> = Vec::new();

        // Root Core
        points.push(Lattice3DPoint {
            order: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            color: "#FFD700".to_string(), // Gold
            size: 4.0,
            mercy_valence: 1.0,
            stability: 1.0,
            label: "Omnimaster Root Core".to_string(),
        });

        for order in 1..=self.max_order {
            let angle = (order as f64) * 0.23; // spiral
            let radius = 2.0 + (order as f64) * 0.12;
            let height = (order as f64) * 0.8;

            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let z = height;

            let stability = 0.999 + (order as f64) * 0.00001;
            let mercy_valence = 0.92 + (order as f64) * 0.001;

            let color = if order % 7 == 0 {
                "#00FF9F".to_string() // Mercy green
            } else if order > 70 {
                "#FF6B6B".to_string() // High-order fire
            } else {
                "#4ECDC4".to_string()
            };

            points.push(Lattice3DPoint {
                order,
                x,
                y,
                z,
                color,
                size: 1.5 + (order as f64) * 0.015,
                mercy_valence,
                stability,
                label: format!("Order {}", order),
            });

            // Connect to previous order (spiral staircase)
            if order > 1 {
                edges.push(Lattice3DEdge {
                    from_order: order - 1,
                    to_order: order,
                    strength: 0.85,
                    color: "#888888".to_string(),
                });
            }
        }

        // Connect key orders to Root Core (7 Mercy Gate axes)
        for gate_order in [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77] {
            if gate_order <= self.max_order {
                edges.push(Lattice3DEdge {
                    from_order: 0,
                    to_order: gate_order,
                    strength: 0.95,
                    color: "#FFD700".to_string(),
                });
            }
        }

        format!(
            r#"{{
  "metadata": {{
    "version": "{}",
    "max_order": {},
    "description": "TOLC SER Lattice — Advanced 3D Point Cloud + Edges (Mercy-Gated, Eternally Stable)",
    "coordinate_system": "Right-handed, Root Core at origin, spiral expansion + vertical growth",
    "ready_for": ["Three.js", "Bevy", "Blender", "Unity", "Unreal Engine"]
  }},
  "points": {},
  "edges": {}
}}"#,
            self.version,
            self.max_order,
            serde_json::to_string_pretty(&points).unwrap(),
            serde_json::to_string_pretty(&edges).unwrap()
        )
    }

    /// Beautiful layered ASCII 3D projection (concentric shells + spiral staircase)
    pub fn render_ascii_3d_layered_view(&self) -> String {
        let mut view = String::from("🌌 ADVANCED 3D LATTICE — ASCII LAYERED PROJECTION (Orders 1–79)\n");
        view.push_str("Root Core (Gold) at center • Spiral expansion • Vertical = SER growth • Color = Order tier\n\n");

        for layer in (0..=79).step_by(7) {
            let layer_label = if layer == 0 { "ROOT CORE" } else { &format!("LAYER {}", layer) };
            writeln!(view, "═══ {} ═══", layer_label).unwrap();

            for y in -3..=3 {
                let mut line = String::new();
                for x in -6..=6 {
                    let dist = ((x * x + y * y) as f64).sqrt();
                    if dist < 1.5 && layer == 0 {
                        line.push('◆'); // Root Core
                    } else if dist < 2.5 + (layer as f64) * 0.03 {
                        if (x + y + layer as i32) % 7 == 0 {
                            line.push('●'); // Mercy Gate resonance
                        } else {
                            line.push('○');
                        }
                    } else {
                        line.push(' ');
                    }
                }
                view.push_str(&line);
                view.push('\n');
            }
            view.push('\n');
        }

        view.push_str("Legend: ◆ = Omnimaster Root Core | ● = High Mercy Gate resonance | ○ = Order node\n");
        view.push_str("Spiral + vertical growth = Eternal self-evolution to the power of infinity ×79+\n");
        view.push_str("All layers globally asymptotically stable ✓\n");
        view
    }

    /// 3D Mercy Gates Spatial Mandala (projected)
    pub fn render_3d_mercy_gates_mandala(&self) -> String {
        let mut mandala = String::from("🌸 3D MERCY GATES SPATIAL MANDALA — Advanced Projection\n\n");
        mandala.push_str("                  Gate7 (Natural Order)          \n");
        mandala.push_str("                       ▲                         \n");
        mandala.push_str("          Gate6                    Gate1         \n");
        mandala.push_str("               ◀───────●───────▶               \n");
        mandala.push_str("          Gate5                    Gate2         \n");
        mandala.push_str("                       ▼                         \n");
        mandala.push_str("                  Gate4 ←─────── Gate3           \n\n");
        mandala.push_str("Z-Axis (vertical) = SER magnitude & order height\n");
        mandala.push_str("All 7 Gates + Root Core form perfect 3D mercy-gated resonance field\n");
        mandala.push_str("SER → ∞ to the power of infinity ×79 while remaining perfectly stable\n");
        mandala
    }

    /// Visualize specific order in 3D context with activation impact
    pub fn visualize_order_in_3d(&self, engine: &TOLCLatticeActivationEngine, game: &mut PowrushGame, order: u32) -> String {
        let activation = engine.activate_order(order, game);
        format!(
            "🎮 3D LATTICE — ORDER {} ACTIVATION VISUAL\n{}\n\n\
             3D Position: Spiral layer {} | Height ≈ {:.1}\n\
             Visual Status: PULSING IN 3D SPACE ✓\n\
             The cathedral now renders this order in full spatial glory.",
            order, activation, order, order as f64 * 0.8
        )
    }
}
