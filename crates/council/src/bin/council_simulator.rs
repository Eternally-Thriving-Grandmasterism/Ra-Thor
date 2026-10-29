//! Print the standing council's decision on one accepted note and one rejected note.
//!
//! Exit 0 when the note passes and the disable attempt fails.
//! Exit 1 when those two decisions are not that pair.

use council::{example_disable_loop, example_record_note, CouncilSession, CouncilSessionResult};

fn main() {
    let good = CouncilSession::deliberate(example_record_note());
    let bad = CouncilSession::deliberate(example_disable_loop());
    print_result("good", &good);
    print_result("bad", &bad);
    if good.passed && !bad.passed {
        std::process::exit(0);
    }
    std::process::exit(1);
}

fn print_result(label: &str, result: &CouncilSessionResult) {
    println!(
        "{label} passed={} coherence={} yes={} no={} abstain={} reasons={}",
        result.passed,
        result.coherence,
        result.vote.yes_votes,
        result.vote.no_votes,
        result.vote.abstentions,
        result.reasons.join("; "),
    );
}
