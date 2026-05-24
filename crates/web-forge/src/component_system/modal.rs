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
}

impl Modal {
    pub fn new(size: ModalSize) -> Self {
        Self { size }
    }
}