use evolution::mercy_public_agi_integration::*;
use evolution::usage_examples::*;

#[tokio::test]
async fn test_phase_b_public_agi_cycle() {
    let proposal = "Integrate dynamic valence for public threads with full TOLC and Sovereignty Gate.";
    let result = example_phase_b_public_agi_cycle(proposal).await;
    assert!(result.contains("AGi acceleration") || result.contains("mercy-filtered"));
}

#[tokio::test]
async fn test_eternal_positive_emotions() {
    let result = example_eternal_positive_emotions(777).await;
    assert!(result.contains("Eternal positive emotions") && result.contains("heaven"));
}