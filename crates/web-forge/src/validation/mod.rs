pub mod html_validator;
pub mod validate;
pub mod component_validator;
pub mod rules;
pub mod accessibility_scorer;

pub use html_validator::HtmlValidator;
pub use accessibility_scorer::{calculate_wcag_aa_score, WcagAaScore};
