use crate::component_system::contract::ComponentContract;

pub struct Modal {
    pub size: ModalSize,
}

#[derive(Debug, Clone, Copy)]
pub enum ModalSize {
    Small,
    Medium,
    Large,
}

impl ComponentContract for Modal {
    fn name(&self) -> &'static str { "Modal" }
    fn requires_token_compliance(&self) -> bool { true }

    /// Custom validation: Modal should have expected size class
    fn validate(&self, html_fragment: &str) -> Vec<String> {
        let mut issues = vec![];

        let expected_class = match self.size {
            ModalSize::Small => "modal-Small",
            ModalSize::Medium => "modal-Medium",
            ModalSize::Large => "modal-Large",
        };

        if !html_fragment.contains(&expected_class) {
            issues.push(format!("Modal is missing expected size class: {}", expected_class));
        }

        issues
    }
}

impl Modal {
    pub fn new(size: ModalSize) -> Self {
        Self { size }
    }
}