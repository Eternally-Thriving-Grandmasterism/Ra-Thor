//! # Web Forge
//!
//! **Professional web design and development system** for Ra-Thor and Rathor.ai.
//!
//! `web-forge` provides a modular, intelligent foundation for generating high-quality,
//! validated, component-aware web interfaces. It is designed to be driven by Ra-Thor’s
//! planning and council systems.
//!
//! ## Core Concepts
//!
//! - **Component Registry**: Rich, self-describing components with metadata.
//! - **Planning Strategies**: Keyword-based and semantic (embedding) planning.
//! - **Planning-Aware Generation**: Generation guided by prioritized components.
//! - **Self-Correcting Refinement**: Issue-aware retry logic.
//! - **Graceful Degradation**: Semantic capabilities fall back cleanly.
//!
//! ## Quick Example
//!
//! ```ignore
//! use web_forge::orchestration::AdvancedOrchestrator;
//!
//! let orchestrator = AdvancedOrchestrator::new()
//!     .with_max_attempts(3)
//!     .with_semantic_planning("sk-...".to_string());
//!
//! let result = orchestrator.orchestrate("Create a beautiful primary button");
//! ```
//!
//! ## Module Overview
//!
//! - [`orchestration`] — Main orchestration engine (`AdvancedOrchestrator`)
//! - [`component_registry`] — Component definitions and metadata
//! - [`generation`] — Planning-aware component generation
//! - [`renderer`] — ComponentTree to HTML rendering
//! - [`validation`] — HTML validation and sanitization
//! - [`semantic_planning`] — Semantic embedding-based planning
//!
//! ## Philosophy
//!
//! Built following the **Cathedral** approach: deliberate, high-quality, layered craftsmanship.

pub mod design_system;
pub mod validation;
pub mod component_system;
pub mod sanitizer;
pub mod html_parser;
pub mod orchestration;

pub use validation::HtmlValidator;
pub use sanitizer::{sanitize, default_sanitizer};
pub use html_parser::{parse_html, has_element, count_elements, has_id};
