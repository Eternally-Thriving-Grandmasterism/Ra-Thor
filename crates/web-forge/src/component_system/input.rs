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
}

impl Input {
    pub fn new(input_type: InputType) -> Self {
        Self { input_type }
    }
}