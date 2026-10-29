//! Record a proposal title only when the session passed.
//!
//! A rejection leaves the log unchanged. This crate does not write into
//! another crate's kernel.

use crate::council_session::CouncilSessionResult;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutcomeLog {
    pub applied_titles: Vec<String>,
}

pub fn apply_outcome(log: &mut OutcomeLog, result: &CouncilSessionResult) {
    if result.passed {
        log.applied_titles.push(result.proposal.title.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::council_session::{example_disable_loop, example_record_note, CouncilSession};

    #[test]
    fn only_a_passed_session_is_recorded() {
        let mut log = OutcomeLog::default();
        let good = CouncilSession::deliberate(example_record_note());
        let bad = CouncilSession::deliberate(example_disable_loop());
        apply_outcome(&mut log, &good);
        apply_outcome(&mut log, &bad);
        assert_eq!(
            log.applied_titles,
            vec!["Record the research note".to_string()]
        );
    }
}
