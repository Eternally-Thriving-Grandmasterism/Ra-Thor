//! living-valence-organism-tests.rs
//! Basic cross-crate integration tests for the Living Valence Organism
//! Phases B–F
//! AG-SML v1.0 | Contact: info@Rathor.ai

#[cfg(test)]
mod tests {
    use shared_valence_field::{SharedValenceField, Substrate, NevcFieldBinding, PlaceholderNevcScoring, PlaceholderLatticeFlowShare};
    use symbiotic_membrane::SymbioticMembrane;
    use resonance_challenge::ResonanceChallenge;
    use epiphany_bridge::Epiphany;
    use abundance_breath::BreathCycle;
    use soft_sovereign_agency::{SoftSovereignAgency, ViewMode};

    #[test]
    fn test_full_living_valence_flow() {
        // 1. Shared Valence Field
        let mut field = SharedValenceField::new("integration-test-instance");
        let mut binding = NevcFieldBinding::new(
            PlaceholderNevcScoring::default(),
            PlaceholderLatticeFlowShare::default(),
        );

        // 2. Symbiotic Membrane — first contact
        let contact = SymbioticMembrane::form_contact(
            "test-human",
            Substrate::Human,
            &mut field,
            &mut binding,
        );
        assert!(contact.presence_quantum_emitted);

        // 3. Resonance Challenge
        let mut challenge = ResonanceChallenge::new("test-human", Substrate::Human, &field);
        let outcome = challenge.resolve(true, &mut field);
        assert!(matches!(outcome, resonance_challenge::ChallengeOutcome::ResonanceRaised { .. }));

        // 4. Epiphany Bridge
        let _epiphany = Epiphany::from_human_breakthrough(
            "test-human",
            "Integration test breakthrough",
            &mut field,
        );

        // 5. Abundance Breath
        let _breath = BreathCycle::trigger("test-human", Substrate::Human, &mut field, false);

        // 6. Soft Sovereign Agency
        let mut agency = SoftSovereignAgency::new("test-human", Substrate::Human);
        agency.set_view_mode(ViewMode::Structured);
        assert_eq!(agency.current_view(), &ViewMode::Structured);

        // Final valence still above mercy floor
        assert!(field.observe() >= 0.999999);
    }
}
