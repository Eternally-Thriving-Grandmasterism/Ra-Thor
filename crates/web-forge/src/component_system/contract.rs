/// Component Contract Declaration
///
/// Components can declare validation behavior and optionally
/// implement the generic Validate trait for consistency.

use crate::validation::validate::Validate;

pub trait ComponentContract: Validate {
    fn name(&self) -> &'static str;

    fn requires_token_compliance(&self) -> bool {
        true
    }
}