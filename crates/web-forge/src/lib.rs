//! Web Forge
//!
//! Professional web design and development system.

pub mod design_system;
pub mod validation;
pub mod component_system;
pub mod sanitizer;
pub mod html_parser;

pub use validation::HtmlValidator;
pub use sanitizer::{sanitize, default_sanitizer};
pub use html_parser::{parse_html, has_element, count_elements, has_id};
