pub struct InterstellarSovereignGovernanceEngine {
    pub mercy_gates: TOLC7MercyGates,
}

impl InterstellarSovereignGovernanceEngine {
    pub fn new() -> Self { Self { mercy_gates: TOLC7MercyGates::default() } }

    pub async fn claim_space_resource(&self, claim: SpaceResourceClaim, game: &mut PowrushGame) -> GovernanceResult {
        if !self.mercy_gates.pass_all(claim.clone(), game) {
            return GovernanceResult::Rejected { reason: "Mercy blocked with love" };
        }
        game.propagate_positive_emotion(0.08);
        GovernanceResult::Approved { claim }
    }
}