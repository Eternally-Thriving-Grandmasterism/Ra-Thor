/// Component Tree (Structured Output Format)
///
/// This defines the advanced JSON structure that generators should target.
/// It allows component-aware, hierarchical, and validated generation.

use serde::{Deserialize, Serialize};

/// A node in the component tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    /// Name of the component (must match ComponentRegistry)
    pub component: String,

    /// Props passed to the component
    #[serde(default)]
    pub props: serde_json::Value,

    /// Child components (for layout/container components)
    #[serde(default)]
    pub children: Vec<ComponentNode>,

    /// Optional text content (for simple components)
    #[serde(default)]
    pub text: Option<String>,
}

/// Root of a generated component tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTree {
    pub root: ComponentNode,
    /// Optional metadata
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

impl ComponentTree {
    pub fn new(root: ComponentNode) -> Self {
        Self { root, metadata: None }
    }
}
