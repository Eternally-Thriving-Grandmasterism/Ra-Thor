//! Web Forge
//!
//! Professional web design and development system.

pub mod design_system;
pub mod validation;
pub mod component_system;
pub mod sanitizer;

pub use validation::HtmlValidator;
pub use sanitizer::{sanitize, default_sanitizer};
