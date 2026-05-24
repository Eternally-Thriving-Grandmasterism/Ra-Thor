/// Component Tree Renderer
///
/// Production-grade renderer that converts `ComponentTree` into HTML.
/// Supports component-specific classes, arbitrary attributes, and basic prop handling.

use crate::orchestration::component_tree::{ComponentNode, ComponentTree};
use serde_json::Value;

/// Renders a full ComponentTree to HTML string.
pub fn render_tree(tree: &ComponentTree) -> String {
    render_node(&tree.root)
}

fn render_node(node: &ComponentNode) -> String {
    let tag = map_component_to_tag(&node.component);
    let attrs = build_attributes(&node.component, &node.props);

    let mut html = if attrs.is_empty() {
        format!("<{}>", tag)
    } else {
        format!("<{} {}>", tag, attrs)
    };

    if let Some(text) = &node.text {
        html.push_str(text);
    }

    for child in &node.children {
        html.push_str(&render_node(child));
    }

    html.push_str(&format!("</{}>", tag));
    html
}

/// Maps component names to HTML tags.
fn map_component_to_tag(component: &str) -> &str {
    match component {
        "Button" => "button",
        "Card" => "div",
        "Input" => "input",
        "Modal" => "div",
        _ => "div",
    }
}

/// Builds attributes and component classes from props.
fn build_attributes(component: &str, props: &Value) -> String {
    let mut parts = vec![];
    let mut classes = vec![];

    if let Value::Object(map) = props {
        for (key, value) in map {
            let val_str = match value {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                _ => value.to_string(),
            };

            match key.as_str() {
                "variant" | "size" | "padding" => {
                    classes.push(format!("{}-{}", component.to_lowercase(), val_str));
                }
                "class" => {
                    classes.push(val_str);
                }
                "id" => {
                    parts.push(format!("id=\"{}\"", val_str));
                }
                _ => {
                    // Escape basic quotes for safety
                    let safe_val = val_str.replace('"', """);
                    parts.push(format!("{}={:?}", key, safe_val));
                }
            }
        }
    }

    if !classes.is_empty() {
        parts.push(format!("class=\"{}\"", classes.join(" ")));
    }

    parts.join(" ")
}
