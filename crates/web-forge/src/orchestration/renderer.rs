/// Component Tree Renderer
///
/// Converts a ComponentTree (JSON structure) into actual HTML.
/// This closes the loop from generation to renderable output.

use crate::orchestration::component_tree::{ComponentNode, ComponentTree};

/// Renders a ComponentTree into HTML.
pub fn render_tree(tree: &ComponentTree) -> String {
    render_node(&tree.root)
}

/// Recursively renders a ComponentNode.
fn render_node(node: &ComponentNode) -> String {
    let tag = map_component_to_tag(&node.component);

    let mut html = format!("<{}>", tag);

    // Add text content if present
    if let Some(text) = &node.text {
        html.push_str(text);
    }

    // Render children
    for child in &node.children {
        html.push_str(&render_node(child));
    }

    html.push_str(&format!("</{}>", tag));
    html
}

/// Maps component names to HTML tags.
/// In a full system, this would use actual component templates.
fn map_component_to_tag(component: &str) -> &str {
    match component {
        "Button" => "button",
        "Card" => "div",
        "Input" => "input",
        "Modal" => "div",
        "div" => "div",
        _ => "div", // fallback
    }
}
