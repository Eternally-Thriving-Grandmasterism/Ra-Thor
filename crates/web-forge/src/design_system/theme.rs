/// Theme Support
///
/// Basic structure for multi-theme support (default, light, future dark/custom).

#[derive(Debug, Clone)]
pub enum Theme {
    Default,
    Light,
    // Future: Dark, Custom(String)
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Default
    }
}