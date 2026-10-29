//! Gated queued-intent face for GitHub tool fires.
//!
//! This is **not** a write. `create_branch` / `update_file` stay unused here.
//! A queued branch intent fails closed if the evidence receipt is missing or
//! the chain is marked broken. `from_sha` must be a commit SHA, not a branch
//! name (same law as `create_branch`).
//!
//! Contact: info@Rathor.ai. Independent of xAI.

use crate::GitHubError;

/// Receipt produced by `lattice-conductor-v14` evidence chain. Opaque strings only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReceipt {
    pub record_hash: String,
    pub previous_hash: String,
    pub chain_intact: bool,
    /// `council` / `wrap` / `tool` (Ra-Thor-native). Tool fires require `tool`.
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedBranchIntent {
    pub branch_name: String,
    pub from_sha: String,
    pub evidence_hash: String,
}

fn is_commit_sha(s: &str) -> bool {
    let n = s.len();
    (n == 40 || n == 64) && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Queue a branch intent. No network. Fail closed without a live tool receipt.
pub fn queued_branch_intent(
    branch_name: &str,
    from_sha: &str,
    receipt: Option<&EvidenceReceipt>,
) -> Result<QueuedBranchIntent, GitHubError> {
    let Some(receipt) = receipt else {
        return Err(GitHubError {
            message: "missing evidence record — no GitHub side effect".into(),
            status: None,
        });
    };
    if !receipt.chain_intact {
        return Err(GitHubError {
            message: "broken evidence chain — GitHub side effect rejected".into(),
            status: None,
        });
    }
    if receipt.record_hash.trim().is_empty() {
        return Err(GitHubError {
            message: "missing evidence record hash — no GitHub side effect".into(),
            status: None,
        });
    }
    if receipt.previous_hash.trim().is_empty() {
        return Err(GitHubError {
            message: "broken evidence chain: empty previous-hash".into(),
            status: None,
        });
    }
    if receipt.kind != "tool" {
        return Err(GitHubError {
            message: format!(
                "queued intent requires kind=tool (got {})",
                receipt.kind
            ),
            status: None,
        });
    }
    if !is_commit_sha(from_sha) {
        return Err(GitHubError {
            message: "create_branch / queued intent needs a real commit SHA, not a branch name"
                .into(),
            status: None,
        });
    }
    if branch_name.trim().is_empty() {
        return Err(GitHubError {
            message: "empty branch name".into(),
            status: None,
        });
    }

    Ok(QueuedBranchIntent {
        branch_name: branch_name.to_string(),
        from_sha: from_sha.to_string(),
        evidence_hash: receipt.record_hash.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_receipt() -> EvidenceReceipt {
        EvidenceReceipt {
            record_hash: "aa".repeat(32),
            previous_hash: "00".repeat(32),
            chain_intact: true,
            kind: "tool".into(),
        }
    }

    #[test]
    fn missing_receipt_fails_side_effect() {
        let sha = "a".repeat(40);
        let err = queued_branch_intent("feat/x", &sha, None).unwrap_err();
        assert!(err.message.contains("missing evidence"));
    }

    #[test]
    fn broken_chain_is_rejected() {
        let sha = "b".repeat(40);
        let mut rec = tool_receipt();
        rec.chain_intact = false;
        let err = queued_branch_intent("feat/x", &sha, Some(&rec)).unwrap_err();
        assert!(err.message.contains("broken evidence chain"));
    }

    #[test]
    fn empty_previous_hash_is_broken() {
        let sha = "c".repeat(40);
        let mut rec = tool_receipt();
        rec.previous_hash.clear();
        let err = queued_branch_intent("feat/x", &sha, Some(&rec)).unwrap_err();
        assert!(err.message.contains("previous-hash"));
    }

    #[test]
    fn branch_name_as_sha_is_rejected() {
        let rec = tool_receipt();
        let err = queued_branch_intent("feat/x", "main", Some(&rec)).unwrap_err();
        assert!(err.message.contains("commit SHA"));
    }

    #[test]
    fn valid_tool_receipt_queues_intent_without_network() {
        let sha = "d".repeat(40);
        let rec = tool_receipt();
        let intent = queued_branch_intent("feat/gated", &sha, Some(&rec)).unwrap();
        assert_eq!(intent.branch_name, "feat/gated");
        assert_eq!(intent.from_sha, sha);
        assert_eq!(intent.evidence_hash, rec.record_hash);
    }
}
