/// Component Contract Declaration
///
/// Truth: Professional components should be able to declare
/// their own validation expectations and provide custom validation logic.

pub trait ComponentContract {
    /// Name of the component
    fn name(&self) -> &'static str;

    /// Whether this component requires token compliance
    fn requires_token_compliance(&self) -> bool {
        true
    }

    /// Custom validation logic provided by the component itself
    /// Default implementation does nothing.
    fn validate(&self, _html_fragment: &str) -> Vec<String> {
        vec![]
    }
}