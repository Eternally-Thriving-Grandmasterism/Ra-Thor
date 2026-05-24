/// Button Component
///
/// Professional button definitions following design tokens.

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

impl Button {
    pub fn new(variant: ButtonVariant, size: ButtonSize) -> Self {
        Self { variant, size }
    }

    pub fn render(&self) -> String {
        // Future: Generate HTML based on tokens
        format!("<button class=\"btn btn-{:?} btn-{:?}\">Button</button>", self.variant, self.size)
    }
}