//! crates/powrush/src/ui_adaptation.rs
//! Basic Adaptive UI scaffolding for Powrush-MMO

use bevy::prelude::*;
use crate::experience_tier::ExperienceTier;

/// Resource that holds current UI scale and layout preferences.
#[derive(Resource, Debug, Clone)]
pub struct UiAdaptation {
    pub scale: f32,
    pub use_compact_layout: bool,
    pub show_advanced_panels: bool,
}

impl Default for UiAdaptation {
    fn default() -> Self {
        Self {
            scale: 1.0,
            use_compact_layout: false,
            show_advanced_panels: true,
        }
    }
}

impl UiAdaptation {
    pub fn from_experience_tier(tier: ExperienceTier) -> Self {
        match tier {
            ExperienceTier::DesktopFull => Self {
                scale: 1.0,
                use_compact_layout: false,
                show_advanced_panels: true,
            },
            ExperienceTier::MobileTown => Self {
                scale: 1.2,
                use_compact_layout: true,
                show_advanced_panels: false,
            },
            ExperienceTier::TabletBalanced => Self {
                scale: 1.1,
                use_compact_layout: false,
                show_advanced_panels: true,
            },
        }
    }
}

pub struct UiAdaptationPlugin;

impl Plugin for UiAdaptationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiAdaptation>();
        app.add_systems(Update, update_ui_adaptation);
    }
}

fn update_ui_adaptation(
    tier: Res<ExperienceTier>,
    mut ui: ResMut<UiAdaptation>,
) {
    let new_ui = UiAdaptation::from_experience_tier(*tier);
    if ui.scale != new_ui.scale || ui.use_compact_layout != new_ui.use_compact_layout {
        *ui = new_ui;
        info!("[Powrush] UI Adaptation updated for tier: {:?}", *tier);
    }
}
