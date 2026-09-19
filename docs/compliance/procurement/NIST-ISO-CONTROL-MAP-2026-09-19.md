# NIST AI RMF x ISO/IEC 42001 control map — honest coverage

**Date:** 2026-09-19  
**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Status:** draft crosswalk. Not a Statement of Applicability. Not a certificate. Not endorsed by NIST or ISO.  
**Fill:** Prompt B after A+E. Tip `88bbdd0b3` (main after #544). Mapped-to-NIST is not complies-with-NIST. No FA% / METR numbers.

Coverage tags (use only these):

| Tag | Meaning |
| --- | --- |
| PRESENT | Named artifact on HEAD; not proof of effectiveness |
| PARTIAL | Artifact exists; gaps published |
| INTENT | Design note or policy draft only |
| NONE | No artifact |
| HOLD | Deliberately not filed / not claimed |

Do not upgrade a tag without a new file + commit. HOLD stays HOLD for ISO application and EU conformity.

---

## NIST AI RMF 1.0 (working language for US buyers)

| Function | Subcategory (representative) | HEAD evidence | Tag |
| --- | --- | --- | --- |
| GOVERN | Policies | `PUBLIC_CLAIM.lock.md`; `docs/compliance/PUBLIC-CLAIM-DISCIPLINE.md`; `docs/compliance/LICENSE-LIVING-STAMP.md` (AG-SML v1.1 living grant) | PRESENT (policy text) |
| GOVERN | Accountability | Sole steward named in the lock; AIMS §2 roles (`docs/compliance/aims/AIMS-SKELETON-2026-08-31.md`); PATSAGi minutes under `docs/science/` including `PATSAGI-COUNCIL-MINUTE-2026-09-19-PROCUREMENT-READINESS.md`. Counsel unnamed. No notified body. | PARTIAL |
| GOVERN | Third-party / suppliers | `docs/compliance/MODEL-MAP-EGRESS-2026-08-31.md` names xAI, X, Anthropic, Google Translate, esm.run / MLC. No vendor no-training / retention letters on HEAD. | PARTIAL |
| MAP | System categorization | `docs/compliance/procurement/SYSTEM-CARD-ADMISSION-SHELL-2026-09-19.md` (S1-L0-ADMIT draft filter); isolate list `docs/compliance/DO-NOT-SHIP-2026-08-31.md`; AIMS §3 inventory | PRESENT (self-class) |
| MAP | Context / data | `docs/compliance/MODEL-MAP-EGRESS-2026-08-31.md` §4 Prompt E static survey on `dc48c6b2e` (`93eb88a57` / #544). Source search of family-site JS/HTML. **No packet capture.** Not “never phones home.” | PARTIAL |
| MAP | Impact | AIMS §4 risk note only. No scored impact / likelihood register file on HEAD. | INTENT |
| MEASURE | Metrics | `docs/GATE_EVAL.md` fixtures + published misses; `docs/compliance/evals/RESULTS-2026-08-31.md` still **UNMEASURED**. Zero is allowed. No invented FA% / METR. | PARTIAL |
| MEASURE | Monitoring | `.github/workflows/mercy-security-tier1.yml`; named `cargo test -p mercy-security`. No production SIEM / WORM store. | PARTIAL |
| MEASURE | Logging | `docs/compliance/procurement/DECISION-RECORD-SCHEMA.md` + `crates/mercy-security/src/decision_record.rs` + `crates/mercy-security/tests/decision_record.rs` (`f4bfedc39` / #543). Replayable admit/block/override JSON; hash of ingest only. **Not a hash-chained SIEM.** Not EU AI Act Article 12 logging. | PARTIAL |
| MANAGE | Risk treatment | `docs/compliance/DO-NOT-SHIP-2026-08-31.md`; HOLD cert language in AIMS + this map; GATE_EVAL published misses. | PARTIAL |
| MANAGE | Incident / rollback | Kill-switch language in `docs/compliance/procurement/PILOT-SOW-AND-LIMITS-MEMO-2026-09-19.md` (draft template, not an executed runbook). | INTENT |
| MANAGE | Third-party | `docs/compliance/NO-CLIENT-SECRETS-2026-08-31.md` procedure exists. No vendor no-training letters. | PARTIAL |

## ISO/IEC 42001:2023 Annex A (what procurement can point to after an auditor)

Application to a registrar: **HOLD**. Not filed. Do not upgrade this HOLD.

| Annex A area | Orientation | HEAD evidence | Tag |
| --- | --- | --- | --- |
| A.2 Policies related to AI | Claim lock + discipline | `PUBLIC_CLAIM.lock.md`; `docs/compliance/PUBLIC-CLAIM-DISCIPLINE.md` | PRESENT (draft policy) |
| A.3 Internal organization | Roles | AIMS §2: operator + PATSAGi councils; no notified body; counsel unnamed | PARTIAL |
| A.4 Resources | Compute, data, people | AIMS §2 one steward; local-first default; `docs/compliance/procurement/HIRE-PARTNER-LATER-PLAN-2026-09-19.md` names the missing on-call / successor roster | PARTIAL |
| A.5 Assessing impacts | AIA / impact | AIMS §4 risk note only. No scored AIA file. | INTENT |
| A.6 AI system life cycle | Change control | Workspace pin 14.15.6 in the lock; `docs/compliance/SITE-VERSION-LOCK-2026-08-31.md`; AIMS §7 no silent bump | PARTIAL |
| A.7 Data for AI systems | Quality / provenance | No training corpus claimed for Layer 0 keywords. No provenance register file. | INTENT |
| A.8 Information for interested parties | Public docs | Root `README.md`; `PUBLIC_CLAIM.lock.md`; `docs/FAQ.md` | PARTIAL |
| A.9 Use of AI systems | HITL | `docs/EMPLOY.md` employ loop ends in human Act; system card §6 | PARTIAL |
| A.10 Third-party / customers | Supply chain | Egress named in the Prompt E map. **No vendor contract / no-training letter file on HEAD.** | NONE (contracts) |

## EU AI Act (only if use case or customer base touches Europe)

| Topic | This surface | Tag |
| --- | --- | --- |
| Classification | Draft filter / ingest gate on the system card — not offered as Annex III high-risk | INTENT (self-class) |
| Logging / post-market | Schema now on HEAD (`DECISION-RECORD-SCHEMA.md` + `DecisionRecord`, #543). Replay of “what the unattended gate said about this hash.” Schema file itself: not Article 12 / Annex IV logging; not a SIEM. | PARTIAL |
| Conformity assessment | Not filed | HOLD |
| privacy.html “compatible with” EU AI Act | Compatibility *language* on the public page; egress map §5 keeps it off any cert pack until counsel redlines | INTENT |

A new use case (hiring, credit, essential services) invalidates this row and needs a new card.

---

## After Prompt A + E (what changed on this fill)

- **MEASURE / logging:** NONE-shaped “schema not yet shipped” → **PARTIAL**. New files: `DECISION-RECORD-SCHEMA.md`, `crates/mercy-security/src/decision_record.rs` (`f4bfedc39`, merge `dc48c6b2e`, #543). Still not PRESENT-as-SIEM.
- **MAP / data-path:** stayed **PARTIAL**. Prompt E filled §4 of `MODEL-MAP-EGRESS-2026-08-31.md` (`93eb88a57`, merge `88bbdd0b3`, #544). Static survey only.
- **ISO application** and **EU conformity:** still **HOLD**. Not filed. Not upgraded.
- **A.10 contracts:** still **NONE**. No contract file on HEAD.

Buyer pack logging row (separate sheet): GAP → PARTIAL for the same reason — schema exists; not a hash-chained SIEM. See `BUYER-EVIDENCE-PACK-2026-09-19.md`.

## What this map is for

- Cursor agents fill evidence path cells only from files that exist.
- Auditors get honest NONE/HOLD instead of slogans.
- Nobody prints mapped-to-NIST as complies-with-NIST.

Capable · Bounded · Corrigible. Inspect ≠ METR.
