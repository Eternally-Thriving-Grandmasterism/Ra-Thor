/// HTML Sanitizer Module
///
/// Uses ammonia to clean potentially unsafe HTML.
/// Provides sensible defaults and configuration options.

use ammonia::{Builder, UrlRelative};

/// Returns a default ammonia Builder with safe, common tags.
pub fn default_sanitizer() -> Builder<'static> {
    let mut builder = Builder::default();

    // Allow common safe tags
    builder
        .add_tags(&[
            "p", "br", "strong", "em", "u", "a", "ul", "ol", "li",
            "h1", "h2", "h3", "h4", "h5", "h6",
            "blockquote", "code", "pre", "hr",
            "table", "thead", "tbody", "tr", "th", "td",
            "img", "figure", "figcaption"
        ])
        .add_generic_attributes(&["class", "id", "style"])
        .add_tag_attributes("a", &["href", "title", "target"])
        .add_tag_attributes("img", &["src", "alt", "title", "width", "height"])
        .url_relative(UrlRelative::PassThrough);

    builder
}

/// Sanitize HTML using the default safe configuration.
pub fn sanitize(html: &str) -> String {
    default_sanitizer().clean(html).to_string()
}
