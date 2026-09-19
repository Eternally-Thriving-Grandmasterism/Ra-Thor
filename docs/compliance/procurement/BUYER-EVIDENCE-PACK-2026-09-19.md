# Buyer evidence pack — how a scoring sheet treats this lattice

**Date:** 2026-09-19  
**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Status:** draft operator pack. Not a vendor questionnaire answer until the operator signs it. Not SOC 2. Not ISO certified.

Typical enterprise stack: general third-party risk **plus** an AI overlay (NIST AI RMF working language; ISO/IEC 42001 if they want a certificate to point at; EU AI Act if the use case or customer base touches Europe).

A striking demo sits last on that stack.

---

## 1. Scoring-sheet order (do not argue with the order)

| Weight band | What they score | Ra-Thor today |
| --- | --- | --- |
| 1. Security + hosting | SOC 2 / ISO 27001, hosting model, pentest, cyber insurance | **Missing.** Local-first research software; no SOC 2 letter in-repo |
| 2. Legal terms | Indemnity, IP, training-data use, exit, license | AG-SML v1.1 living grant + commercial exhibit drafts. Counsel unsigned |
| 3. Operational resilience | On-call, SLA, version pin, rollback, successor | Workspace pin 14.15.6 is real. No SLA. One steward |
| 4. Model governance evidence | System card, data-flow, logs, HITL, measurement | This folder + GATE_EVAL + egress map. Evals UNMEASURED |
| 5. Capability | Demo quality | Last. Do not lead with worldview |

That is why a buyer can need a control layer and still treat a sole steward as an uninsurable dependency. They are not denying the need.

---

## 2. Good-enough artifacts — status

| Artifact | Buyer wants | HEAD / this pack | Tag |
| --- | --- | --- | --- |
| System card | Intended use, out-of-scope, failure modes | `SYSTEM-CARD-ADMISSION-SHELL-2026-09-19.md` | DRAFT |
| Data-flow diagram | On-device vs wrap vs third-party model | `MODEL-MAP-EGRESS-2026-08-31.md` + system card §4 | DRAFT inventory |
| Control map | NIST functions or ISO 42001 Annex A | `NIST-ISO-CONTROL-MAP-2026-09-19.md` | PARTIAL / NONE tagged |
| Logging | Who admitted, blocked, overrode + retention | `DECISION-RECORD-SCHEMA.md` + `mercy-security::DecisionRecord` (#543). Schema exists; not a hash-chained SIEM. | PARTIAL |
| HITL + rollback | Procedure | System card §6–7; employ loop ends in human Act | DRAFT |
| Pilot SOW + kill switch + limits memo | Time-boxed, written limits | `PILOT-SOW-AND-LIMITS-MEMO-2026-09-19.md` | DRAFT |
| Key-person + escrow | Buyer not married to one operator | `HIRE-PARTNER-LATER-PLAN-2026-09-19.md` | GAP / later |
| SOC 2 Type II | Auditor letter | None | NONE |
| Cyber insurance certificate | Named limits | None in-repo | NONE |
| Audited financials | Going-concern | None in-repo | NONE |
| ISO 42001 certificate | Accredited auditor | HOLD application. AIMS skeleton only | HOLD |
| METR-style measurement | Independent time-horizon | Explicitly not claimed | NONE |

Logging row after Prompt A: GAP → PARTIAL because the schema file exists; it is not a hash-chained SIEM.

---

## 3. How to answer an RFP without lying

Allowed sentences:

- Inspectable research software, workspace 14.15.6, Autonomicity Games Inc.
- Admission shell with published fixtures and known misses.
- Optional Grok session the human opens. Not affiliated with xAI.
- Draft filter / ingest gate. Not a high-risk employment or credit system.
- ISO/IEC 42001 is a planning track. We have not applied.

Forbidden sentences:

- We are ISO 42001 / SOC 2 / EU AI Act conformant.
- Mercy gates satisfy your high-risk duties.
- Inspectable repo = audit letter.
- Combined AGSi is demonstrated.
- The wrap never phones home (unless that path is measured).

---

## 4. Practical path (same as the buyer note)

1. Keep the use case narrow (this system card only).
2. Sell, if ever, a time-boxed pilot that leaves artifacts — not a worldview.
3. Accept that ISO / SOC / insurance come before scale.
4. Until those exist, the honest public record is what it already is: research software.
