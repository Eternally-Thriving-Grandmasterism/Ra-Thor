# ActionGovernor fixture proofs

These scenarios document **governor-trips-first** behavior for safe agent runtimes under domain `ContainmentProfile`s.

| Scenario | Profile | Expected |
|----------|---------|----------|
| Rate overflow | `education` (30 actions/min) | 30 benign local OK → 31st `ActionLimitExceeded` |
| Sandbox churn | `education` (max 2 concurrent) | s1+s2 OK → s3 `ActionLimitExceeded` |
| Harm language | any | `HarmRefusalActive` **before** governor increments trips |
| External network | `research` | `ContainmentViolation` before governor trip |
| Long-lived token | `enterprise` | TTL > 900s → `SecretIsolationViolation` |

Executable proofs: `cargo test -p mercy-security safe_agent_runtime`

Contact: **info@Rathor.ai**
