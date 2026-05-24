/// Component Contract Declaration
///
/// Truth: Professional components should be able to declare
/// their own validation expectations and constraints.
///
/// This allows the Validation Engine to understand component
/// contracts and enforce them during generation and editing.

pub trait ComponentContract {
    /// Name of the component
    fn name(&self) -> &'static str;

    /// List of validation rules this component cares about
    fn required_rules(&self) -> Vec<&'static str> {
        vec![]
    }

    /// Whether this component requires token compliance
    fn requires_token_compliance(&self) -> bool {
        true
    }
}