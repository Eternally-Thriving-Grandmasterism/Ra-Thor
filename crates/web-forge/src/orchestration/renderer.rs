/// Component Tree Renderer
///
/// Production-grade HTML renderer with:
/// - Proper void element handling
/// - Boolean attribute support
/// - Strong escaping for attributes and text
/// - aria-* and data-* attribute support
/// - Component-specific class generation

use crate::orchestration::component_tree::{ComponentNode, ComponentTree};
use serde_json::Value;

/// HTML void elements (self-closing).
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
];

/// Boolean attributes that should render without values when true.
const BOOLEAN_ATTRIBUTES: &[&str] = &[
    "disabled", "checked", "selected", "hidden", "readonly", "required",
    "multiple", "autofocus", "autoplay", "controls", "loop", "muted",
];

pub fn render_tree(tree: &ComponentTree) -> String {
    render_node(&tree.root)
}

fn render_node(node: &ComponentNode) -> String {
    let tag = map_component_to_tag(&node.component);
    let attrs = build_attributes(&node.component, &node.props);
    let is_void = VOID_ELEMENTS.contains(&tag);

    let mut html = String::new();

    if attrs.is_empty() {
        html.push_str(if is_void { &format!("<{} />", tag) } else { &format!("<{}>", tag) });
    } else if is_void {
        html.push_str(&format!("<{} {} />", tag, attrs));
    } else {
        html.push_str(&format!("<{} {}>", tag, attrs));
    }

    if let Some(text) = &node.text {
        html.push_str(&escape_text(text));
    }

    for child in &node.children {
        html.push_str(&render_node(child));
    }

    if !is_void {
        html.push_str(&format!("</{}>", tag));
    }

    html
}

fn map_component_to_tag(component: &str) -> &str {
    match component {
        "Button" => "button",
        "Card" => "div",
        "Input" => "input",
        "Modal" => "div",
        "Image" => "img",
        "Link" => "a",
        "Paragraph" => "p",
        "Heading" => "h1",
        _ => "div",
    }
}

fn build_attributes(component: &str, props: &Value) -> String {
    let mut parts = vec![];
    let mut classes = vec![];

    if let Value::Object(map) = props {
        for (key, value) in map {
            let key_str = key.as_str();

            // Boolean attributes
            if BOOLEAN_ATTRIBUTES.contains(&key_str) {
                if value.as_bool().unwrap_or(false) {
                    parts.push(key_str.to_string());
                }
                continue;
            }

            let val_str = match value {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => value.to_string(),
            };

            match key_str {
                "variant" | "size" | "padding" => {
                    classes.push(format!("{}-{}", component.to_lowercase(), val_str));
                }
                "class" => classes.push(val_str),
                "id" | "style" => {
                    parts.push(format!("{}={}", key_str, quote_attribute(&val_str)));
                }
                _ if key_str.starts_with("aria-") || key_str.starts_with("data-") => {
                    parts.push(format!("{}={}", key_str, quote_attribute(&val_str)));
                }
                _ => {
                    parts.push(format!("{}={}", key_str, quote_attribute(&val_str)));
                }
            }
        }
    }

    if !classes.is_empty() {
        parts.push(format!("class=\"{}\"", classes.join(" ")));
    }

    parts.join(" ")
}

/// Escapes text content for safe HTML.
fn escape_text(text: &str) -> String {
    text.replace('&', "&")
        .replace('<', "<")
        .replace('>', ">")
}

/// Escapes and quotes an attribute value.
fn quote_attribute(value: &str) -> String {
    format!("\"{}\"", value
        .replace('&', "&")
        .replace('"', """)
        .replace('<', "<")
        .replace('>', ">"))
}
