/// Generic Validation Trait
///
/// Provides a reusable interface for anything that can be validated.
/// This makes the validation system more generic and extensible.

pub trait Validate {
    /// Run validation and return a list of issues found.
    fn validate(&self) -> Vec<String>;

    /// Returns true if there are no validation issues.
    fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}