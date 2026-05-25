/// Component Tree Renderer
///
/// Production-grade renderer that converts `ComponentTree` into HTML.
/// Features:
/// - Proper handling of void elements (self-closing tags)
/// - Improved attribute escaping
/// - Support for common props (class, id, style, data-*)
/// - Component-specific class generation

use crate::orchestration::component_tree::{ComponentNode, ComponentTree};
use serde_json::Value;

/// Set of HTML void elements that should be self-closing.
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
];

/// Renders a full ComponentTree to HTML string.
pub fn render_tree(tree: &ComponentTree) -> String {
    render_node(&tree.root)
}

fn render_node(node: &ComponentNode) -> String {
    let tag = map_component_to_tag(&node.component);
    let attrs = build_attributes(&node.component, &node.props);
    let is_void = VOID_ELEMENTS.contains(&tag);

    let mut html = if attrs.is_empty() {
        if is_void {
            format!("<{} />", tag)
        } else {
            format!("<{}>", tag)
        }
    } else if is_void {
        format!("<{} {} />", tag, attrs)
    } else {
        format!("<{} {}>", tag, attrs)
    };

    if let Some(text) = &node.text {
        html.push_str(text);
    }

    for child in &node.children {
        html.push_str(&render_node(child));
    }

    if !is_void {
        html.push_str(&format!("</{}>", tag));
    }

    html
}

/// Maps known component names to HTML tags.
fn map_component_to_tag(component: &str) -> &str {
    match component {
        "Button" => "button",
        "Card" => "div",
        "Input" => "input",
        "Modal" => "div",
        "Image" => "img",
        "Link" => "a",
        _ => "div",
    }
}

/// Builds HTML attributes from props with proper escaping.
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
                "class" => classes.push(val_str),
                "id" => parts.push(format!("id=\"{}\"", escape_attribute(&val_str))),
                "style" => parts.push(format!("style=\"{}\"", escape_attribute(&val_str))),
                _ if key.starts_with("data-") => {
                    parts.push(format!("{}={}", key, quote_attribute(&val_str)));
                }
                _ => {
                    parts.push(format!("{}={}", key, quote_attribute(&val_str)));
                }
            }
        }
    }

    if !classes.is_empty() {
        parts.push(format!("class=\"{}\"", classes.join(" ")));
    }

    parts.join(" ")
}

/// Escapes special characters in attribute values.
fn escape_attribute(value: &str) -> String {
    value
        .replace('&', "&")
        .replace('"', """)
        .replace('<', "<")
        .replace('>', ">")
}

/// Wraps a value in quotes with basic escaping.
fn quote_attribute(value: &str) -> String {
    format!("\"{}\"", escape_attribute(value))
}
