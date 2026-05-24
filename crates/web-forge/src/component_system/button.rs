use crate::component_system::contract::ComponentContract;

/// Button Component
///
/// Implements ComponentContract with custom validation logic.

pub struct Button {
    pub variant: ButtonVariant,
    pub size: ButtonSize,
}

#[derive(Debug, Clone, Copy)]
pub enum ButtonVariant {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, Copy)]
pub enum ButtonSize {
    Small,
    Medium,
    Large,
}

impl ComponentContract for Button {
    fn name(&self) -> &'static str {
        "Button"
    }

    fn requires_token_compliance(&self) -> bool {
        true
    }

    /// Custom validation: Button should have a valid variant class
    fn validate(&self, html_fragment: &str) -> Vec<String> {
        let mut issues = vec![];

        if !html_fragment.contains("btn-Primary") && !html_fragment.contains("btn-Secondary") {
            issues.push("Button is missing a valid variant class (btn-Primary or btn-Secondary)".to_string());
        }

        issues
    }
}

impl Button {
    pub fn new(variant: ButtonVariant, size: ButtonSize) -> Self {
        Self { variant, size }
    }
}