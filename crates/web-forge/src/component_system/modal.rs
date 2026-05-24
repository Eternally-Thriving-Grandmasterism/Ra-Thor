/// Modal Component
///
/// Professional modal/dialog component.

pub struct Modal {
    pub size: ModalSize,
}

#[derive(Debug, Clone, Copy)]
pub enum ModalSize {
    Small,
    Medium,
    Large,
}

impl Modal {
    pub fn new(size: ModalSize) -> Self {
        Self { size }
    }

    pub fn render(&self) -> String {
        format!("<div class=\"modal modal-{:?}\">Modal Content</div>", self.size)
    }
}