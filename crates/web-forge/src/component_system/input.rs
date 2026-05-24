/// Input Component
///
/// Professional form input following design tokens.

pub struct Input {
    pub input_type: InputType,
}

#[derive(Debug, Clone, Copy)]
pub enum InputType {
    Text,
    Email,
    Password,
}

impl Input {
    pub fn new(input_type: InputType) -> Self {
        Self { input_type }
    }

    pub fn render(&self) -> String {
        format!("<input type=\"{:?}\" class=\"input\" placeholder=\"Enter value...\" />", self.input_type)
    }
}