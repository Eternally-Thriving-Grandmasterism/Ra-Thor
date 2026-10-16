//! council — a deterministic vote on one proposal.
//!
//! Three standing members weigh token overlap with the proposal text.
//! A majority of yes votes can pass. An empty proposal, a cosmic-loop
//! disable attempt, or a layer 0 disable attempt is rejected, and the
//! vote cannot turn that rejection into an apply.
//!
//! This crate does not call lattice-conductor-v14, kernel, or mercy.
//! Those crates are outside this standalone workspace.
//! Contact: info@Rathor.ai

pub mod coherence;
pub mod council_session;
pub mod deliberation;
pub mod lattice_v14_guard;
pub mod member_profiles;
pub mod outcome_applicator;
pub mod tolc;
pub mod voting;

pub use coherence::compute_session_coherence;
pub use council_session::{
    example_disable_loop, example_record_note, CouncilProposal, CouncilSession,
    CouncilSessionResult,
};
pub use deliberation::{MemberOpinion, MemberVote};
pub use lattice_v14_guard::{
    ensure_cosmic_loop_for_session, proposal_respects_cosmic_loop, proposal_respects_layer0,
};
pub use member_profiles::{standing_profiles, CouncilMemberProfile};
pub use outcome_applicator::{apply_outcome, OutcomeLog};
pub use tolc::{calculate_tolc_resonance, TOLC_8};
pub use voting::{conduct_voting, VoteResult};
