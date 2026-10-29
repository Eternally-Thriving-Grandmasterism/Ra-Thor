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

    /// Custom validation: Card should use proper padding class
    fn validate(&self, html_fragment: &str) -> Vec<String> {
        let mut issues = vec![];

        let expected_class = match self.padding {
            CardPadding::Small => "card-Small",
            CardPadding::Medium => "card-Medium",
            CardPadding::Large => "card-Large",
        };

        if !html_fragment.contains(&expected_class) {
            issues.push(format!("Card is missing expected padding class: {}", expected_class));
        }

        issues
    }
}

impl Card {
    pub fn new(padding: CardPadding) -> Self {
        Self { padding }
    }
}