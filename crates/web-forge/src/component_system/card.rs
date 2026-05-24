/// Card Component
///
/// Professional card container following design tokens.

pub struct Card {
    pub padding: CardPadding,
}

#[derive(Debug, Clone, Copy)]
pub enum CardPadding {
    Small,
    Medium,
    Large,
}

impl Card {
    pub fn new(padding: CardPadding) -> Self {
        Self { padding }
    }

    pub fn render(&self) -> String {
        format!("<div class=\"card card-{:?}\">Card Content</div>", self.padding)
    }
}