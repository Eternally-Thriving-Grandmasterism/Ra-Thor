use crate::component_system::contract::ComponentContract;

pub struct Input {
    pub input_type: InputType,
}

#[derive(Debug, Clone, Copy)]
pub enum InputType {
    Text,
    Email,
    Password,
}

impl ComponentContract for Input {
    fn name(&self) -> &'static str { "Input" }
    fn requires_token_compliance(&self) -> bool { true }

    /// Custom validation: Input should have correct type attribute
    fn validate(&self, html_fragment: &str) -> Vec<String> {
        let mut issues = vec![];

        let expected_type = match self.input_type {
            InputType::Text => "text",
            InputType::Email => "email",
            InputType::Password => "password",
        };

        if !html_fragment.contains(&format!("type=\"{}\"", expected_type)) {
            issues.push(format!("Input is missing expected type attribute: {}", expected_type));
        }

        issues
    }
}

impl Input {
    pub fn new(input_type: InputType) -> Self {
        Self { input_type }
    }
}