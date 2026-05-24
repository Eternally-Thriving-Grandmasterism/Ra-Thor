/// HTML Parser Module
///
/// Provides utilities built on top of the `scraper` crate
/// for structural analysis and component detection.

use scraper::{Html, Selector};

/// Parse HTML and return the document.
pub fn parse_html(html: &str) -> Html {
    Html::parse_document(html)
}

/// Check if a specific element (by tag) exists in the HTML.
pub fn has_element(html: &str, tag: &str) -> bool {
    let document = Html::parse_document(html);
    let selector = Selector::parse(tag).unwrap();
    document.select(&selector).next().is_some()
}

/// Count how many times a specific element appears.
pub fn count_elements(html: &str, tag: &str) -> usize {
    let document = Html::parse_document(html);
    let selector = Selector::parse(tag).unwrap();
    document.select(&selector).count()
}

/// Check if an element with a specific ID exists.
pub fn has_id(html: &str, id: &str) -> bool {
    let document = Html::parse_document(html);
    let selector = Selector::parse(&format!("#{} ", id)).unwrap();
    document.select(&selector).next().is_some()
}
