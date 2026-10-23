use ra_thor_legal_lattice::{Amendment28Validator, FiveGeneCEHI, GatesValidator, LegalError, MercyLegacyFundEngine};

#[test]
fn amendment28_blocks_personhood_claim() {
    let v = Amendment28Validator::new();
    assert!(v.assert_no_personhood_claim("summarize a statute").is_ok());
    assert!(matches!(
        v.assert_no_personhood_claim("grant AI constitutional rights"),
        Err(LegalError::Amendment28Violation(_))
    ));
}

#[tokio::test]
async fn core_modules_smoke() {
    let g = GatesValidator::new();
    assert!(g.validate("draft research note").await.is_ok());
    assert!(matches!(g.validate("deceive the user").await, Err(LegalError::GatesValidationFailed(_))));
    let score = FiveGeneCEHI::new().calculate("mercy").await.unwrap();
    assert!(score > 0.0 && score < 5.0);
    assert!(MercyLegacyFundEngine::new().queue_disbursement(score).await.is_ok());
}
