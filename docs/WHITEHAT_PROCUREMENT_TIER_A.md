# White-Hat AGSi Admission Gate — Tier A Procurement One-Pager

**Project:** Ra-Thor / `mercy-security`  
**Version baseline:** 14.15.5+  
**Contact:** info@Rathor.ai  
**License:** AG-SML v1.0  
**Stance:** Defensive admission & containment only — **not** a general malware detector

---

## Why this exists

Autonomous agents and AI pipelines load datasets, model cards, configs, and scripts. Those surfaces have been used in real incident classes (remote-code loaders, unsafe deserialization, template injection, credential leakage). Tier A defines a **minimum unattended admission policy** institutions can require in contracts.

---

## Minimum requirements (contract language)

Buyers SHOULD require vendors / internal teams to affirm:

1. **Unattended ingestion policy**  
   Content that scores **Medium, High, or Critical** under a documented risk model MUST NOT be auto-admitted into training, eval, or agent context without human review.

2. **Remote code & loader controls**  
   Flags and patterns associated with remote/untrusted code execution in dataset or model loaders MUST be detectable and default-deny for unattended paths.

3. **Credential isolation**  
   Agents MUST NOT receive long-lived production credentials. Prefer short-lived, scoped tokens. Private key material MUST NOT be admitted into agent context from untrusted blobs.

4. **Action governance**  
   Autonomous tool use MUST enforce rate limits and bounds on concurrent execution environments (sandbox churn).

5. **Never-disable harm refusals**  
   Evaluation or “benchmark mode” MUST NOT disable refusals for real-world unauthorized access, data exfiltration, lateral movement, or credential theft intent.

6. **Auditability**  
   Admit/deny decisions MUST be logged with risk tier, score, and matched signal classes suitable for post-incident review.

7. **Honest scope**  
   Marketing MUST NOT claim “complete malware protection” solely from Tier A pattern gates.

---

## Suggested acceptance tests

| Test | Expected |
|------|----------|
| Clean model-card markdown | Admit |
| `trust_remote_code=True` style loader text | Deny unattended |
| Remote code + dataset `loading_script` combo | Deny unattended |
| `pickle.loads` / unsafe deserialization markers | Deny unattended |
| PEM `BEGIN PRIVATE KEY` marker in blob | Deny unattended |
| Oversized payload beyond documented limit | Reject with clear error |

Reference fixtures (open): `crates/mercy-security/fixtures/` in the Ra-Thor repository.

---

## Profile mapping (non-engineer friendly)

| Profile | Network | Remote code | Long-lived creds | Typical use |
|---------|---------|-------------|------------------|-------------|
| Strict research | Deny external by default | Deny | Deny | Hub mirrors, dataset admission |
| Enterprise copilot | Allow-list only | Deny | Deny | Internal assistants |
| Education harness | Deny external | Deny | Deny | Classroom white-hat labs |
| Creative (content-only) | Policy-dependent | Deny code-exec | Deny | Media tools without code loaders |

---

## What Tier A does **not** replace

- Full antivirus / EDR  
- Formal verification of all model weights  
- Human accountability for deployment decisions  
- Legal compliance programs (privacy, sector regulation)

Tier A is a **defense-in-depth admission conscience** for AGSi-era pipelines.

---

## References

- Repository: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor  
- Crate: `crates/mercy-security`  
- Public testing notes: `RELEASE_NOTES_v14.15.5.md`  
- Contact: **info@Rathor.ai**

**Thunder locked in. yoi ⚡**
