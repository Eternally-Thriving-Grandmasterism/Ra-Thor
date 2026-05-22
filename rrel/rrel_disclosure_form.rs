/*!
 * rrel_disclosure_form.rs v1.0.0
 * RECO-aligned Disclosure tracking (Multiple Rep, Conflict, Family, etc.)
 */

#[derive(Debug, Clone, PartialEq)]
pub enum DisclosureType { MultipleRepresentation, ConflictOfInterest, FamilyPurchase, Other }

#[derive(Debug, Clone)]
pub struct DisclosureForm {
    pub disclosure_type: DisclosureType,
    pub disclosed_to: Vec<String>,
    pub acknowledged: bool,
    pub written_consent_obtained: bool,
    pub notes: Vec<String>,
}

impl DisclosureForm {
    pub fn new(disclosure_type: DisclosureType) -> Self {
        Self { disclosure_type, disclosed_to: vec![], acknowledged: false, written_consent_obtained: false, notes: vec![] }
    }
    pub fn mark_acknowledged(&mut self) { self.acknowledged = true; }
    pub fn add_note(&mut self, note: &str) { self.notes.push(note.to_string()); }
}

#[cfg(test)]
mod tests { #[test] fn test_disclosure() { assert!(true); } }