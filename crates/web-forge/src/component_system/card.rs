use crate::component_system::contract::ComponentContract;

/// Card Component

pub struct Card {
    pub padding: CardPadding,
}

#[derive(Debug, Clone, Copy)]
pub enum CardPadding {
    Small,
    Medium,
    Large,
}

impl ComponentContract for Card {
    fn name(&self) -> &'static str { "Card" }
    fn requires_token_compliance(&self) -> bool { true }
}

impl Card {
    pub fn new(padding: CardPadding) -> Self {
        Self { padding }
    }
}