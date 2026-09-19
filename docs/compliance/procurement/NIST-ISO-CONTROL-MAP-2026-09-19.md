# NIST AI RMF x ISO/IEC 42001 control map — honest coverage

**Date:** 2026-09-19  
**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Status:** draft crosswalk. Not a Statement of Applicability. Not a certificate. Not endorsed by NIST or ISO.

Coverage tags (use only these):

| Tag | Meaning |
| --- | --- |
| PRESENT | Named artifact on HEAD; not proof of effectiveness |
| PARTIAL | Artifact exists; gaps published |
| INTENT | Design note or policy draft only |
| NONE | No artifact |
| HOLD | Deliberately not filed / not claimed |

Do not upgrade a tag without a new file + commit.

---

## NIST AI RMF 1.0 (working language for US buyers)

| Function | Subcategory (representative) | HEAD evidence | Tag |
| --- | --- | --- | --- |
| GOVERN | Policies | `PUBLIC_CLAIM.lock.md`, `PUBLIC-CLAIM-DISCIPLINE.md`, AG-SML v1.1 | PRESENT (policy text) |
| GOVERN | Accountability | Sole operator named; PATSAGi minutes; counsel unnamed | PARTIAL |
| GOVERN | Third-party / suppliers | Egress map names xAI, X, Anthropic, CDNs | PARTIAL |
| MAP | System categorization | This pack: S1-L0-ADMIT draft filter; isolate list for high-risk labels | PRESENT (self-class) |
| MAP | Context / data | `MODEL-MAP-EGRESS-2026-08-31.md` | PARTIAL |
| MAP | Impact | AIMS skeleton risk note; no scored register | INTENT |
| MEASURE | Metrics | GATE_EVAL fixtures; evals RESULTS UNMEASURED | PARTIAL / zero allowed |
| MEASURE | Monitoring | No production SIEM; crate tests + CI for mercy-security | PARTIAL |
| MANAGE | Risk treatment | Do-not-ship isolate; HOLD cert; published misses | PARTIAL |
| MANAGE | Incident / rollback | Kill-switch language in pilot SOW draft | INTENT |
| MANAGE | Third-party | No-secrets procedure; no vendor no-training letters | PARTIAL |

## ISO/IEC 42001:2023 Annex A (what procurement can point to after an auditor)

Application to a registrar: **HOLD**.

| Annex A area | Orientation | HEAD evidence | Tag |
| --- | --- | --- | --- |
| A.2 Policies related to AI | Claim lock + discipline | Files exist | PRESENT (draft policy) |
| A.3 Internal organization | Roles | Operator + councils; no notified body; counsel unnamed | PARTIAL |
| A.4 Resources | Compute, data, people | One steward; local-first; no on-call roster | PARTIAL |
| A.5 Assessing impacts | AIA / impact | AIMS risk note only | INTENT |
| A.6 AI system life cycle | Change control | Workspace pin; no silent bump rule | PARTIAL |
| A.7 Data for AI systems | Quality / provenance | No training corpus claimed for Layer 0 keywords | INTENT |
| A.8 Information for interested parties | Public docs | README + lock + FAQ | PARTIAL |
| A.9 Use of AI systems | HITL | Employ loop ends in human Act | PARTIAL |
| A.10 Third-party / customers | Supply chain | Egress named; contracts missing | NONE (contracts) |

## EU AI Act (only if use case or customer base touches Europe)

| Topic | This surface | Tag |
| --- | --- | --- |
| Classification | Draft filter / ingest gate — not offered as Annex III high-risk | INTENT / self-class |
| Logging / post-market | Decision-record schema not yet shipped | NONE |
| Conformity assessment | Not filed | HOLD |
| privacy.html compatible-with EU AI Act | Compatibility language, not conformity | Watch — counsel |

A new use case (hiring, credit, essential services) invalidates this row and needs a new card.

## What this map is for

- Cursor agents fill evidence path cells only from files that exist.
- Auditors get honest NONE/HOLD instead of slogans.
- Nobody prints mapped-to-NIST as complies-with-NIST.
