# PATSAGi Council minute — Procurement readiness for high-stakes buyers

**Date:** 2026-09-19  
**Resolution ID:** `2026-09-19-Procurement-Readiness-DecideAndProceed`  
**Authority:** Permanent PATSAGi Councils **under** TOLC 8  
**Conductor:** sequences only; does not replace gates or councils  
**Workspace:** 14.15.6 (do not bump)  
**Contact:** info@Rathor.ai  
**Parent locks:** [`PUBLIC_CLAIM.lock.md`](../../PUBLIC_CLAIM.lock.md) · [`docs/compliance/PUBLIC-CLAIM-DISCIPLINE.md`](../compliance/PUBLIC-CLAIM-DISCIPLINE.md) · [`docs/compliance/DO-NOT-SHIP-2026-08-31.md`](../compliance/DO-NOT-SHIP-2026-08-31.md) · [`docs/compliance/aims/AIMS-SKELETON-2026-08-31.md`](../compliance/aims/AIMS-SKELETON-2026-08-31.md)

**Status:** inspectable research software. Optional Grok session. Independent of xAI. Not certified. Not a legal product. Combined AGSi stays SURMISE. Inspect ≠ METR. ISO/IEC 42001 application remains **HOLD**.

Trigger: operator asked councils to prepare Ra-Thor for public service to high-stakes organizations, using a buyer-side procurement note (NIST AI RMF / ISO 42001 / EU AI Act overlay; four kill-buckets; evidence-in-buyer-format). Goal: Cursor-agent work now; hire / partner / certify later.

---

## Live facts (HEAD `97a6af38` at branch cut)

- Default members: 12 Core Tier-1 + `mercy-security`. Cargo.toml pins workspace **14.15.6**.
- Public claim lock already forbids ISO 42001 certified, EU AI Act conformant, AGSi-warrantied, xAI affiliation.
- Existing artifacts that *help* a buyer but are not buyer-format:
  - Model/egress map: `docs/compliance/MODEL-MAP-EGRESS-2026-08-31.md`
  - AIMS skeleton (planning only): `docs/compliance/aims/`
  - Eval harness spec, RESULTS UNMEASURED: `docs/compliance/evals/`
  - Admission-shell evidence: `docs/GATE_EVAL.md`, `docs/EVAL_SPEC.md`, `crates/mercy-security`
  - Layer 0 boundary: `docs/LAYER_0_RUNTIME_BOUNDARY.md`
  - Inspect ≠ METR: `docs/MODEL_INSPECT_NOT_METR.md`
- Missing in buyer language: system card, control map with coverage tags, replayable admit/block/override log schema, pilot SOW + kill switch + limits memo, key-person/escrow plan, SOC 2 / cyber insurance / audited financials / successor.
- Open PRs/issues at last survey: treat as empty unless the board says otherwise. Do not invent booked revenue.

---

## Council decision (TOLC 8)

| Approve | Reject / HOLD |
|---|---|
| Name **one** sellable-later surface: Layer 0 **admission shell + decision record** for draft / research ingest | Sell Combined AGSi, mercy gates as a hiring/credit/infra product, legal-lattice, predictive-policing crates |
| Classify that surface as **not high-risk** only while it stays a draft filter / ingest gate with human Act | Classify as EU AI Act high-risk, ISO-certified, or METR-evaluated |
| Land buyer-format *drafts* under `docs/compliance/procurement/` | File ISO 42001, claim SOC 2, invent eval scores |
| Cursor agents may implement **logging schema + replay tests + control-map fill from existing files** | Re-add research forest to `members`; unpark `self-evolution`; bump workspace |
| Keep 12-member Core as sole compile set | Treat compile-green as live safety |
| Plan hire / partner / insurance / escrow as **later** work with named roles | Pretend a sole steward is already an insurable vendor |

Narrow intended use (the only use this pack authorizes agents to document):

> An inspectable **admission shell** that admits or blocks unattended ingest on apply-class paths that cross `MercyGatedApi::handle_request`, plus a replayable decision record (admit / block / override). Optional model wrap is a **named router**, not the product. Human reviews drafts before filing, sale, or public legal claims.

Out of scope as product (already isolate): employment screening, credit, essential services, infrastructure control, law enforcement, legal advice, national accounts, wages, METR time-horizon claims.

---

## Four kill-buckets — honest score

| Bucket | Buyer ask | HEAD truth | Now | Later |
|---|---|---|---|---|
| Counterparty | Solvent vendor, SOC 2 Type II, cyber insurance, successor | One steward; no SOC 2; no insurance cert in-repo; no escrow | Name the gap; draft key-person / escrow memo | Counsel, insurer, successor director, SOC 2 Type I→II |
| Data-path | Who sees prompts; wrap vs on-device; named router in contract | Egress map exists; `call_claude` is live-capable shape; Grok is user-opened session | Keep map current; forbid client secrets; name router in pilot SOW | Contractual no-training / retention terms per vendor |
| Assurance | Third party can replay admit/refuse/override | Keyword fixtures + GATE_EVAL; no hash-chained audit export; evals UNMEASURED | Decision-record schema + replay test; do not invent scores | Independent lab / METR-class only if scoped; ISO control evidence |
| Use-case | Duties follow the job | Mercy language is general | Lock the draft-filter use case in the system card | New card per new use case; never reuse this card for hiring/credit |

---

## What Cursor may touch this tick

Allowed:

- New files under `docs/compliance/procurement/`
- Additive tests in `crates/mercy-security` for a decision-record JSON schema **if** they do not change admit/block policy
- Pointers from `docs/compliance/README.md` and `docs/GATE_EVAL.md` (one-line index only)

Forbidden:

- Workspace version bump
- `[workspace].members` edits
- Public claim inflation
- Filling RESULTS tables with invented numbers
- Shipping `ai-bridge::call_claude` as an offer
- COEP / i18n / forest / Powrush bind

---

## Next honest move

1. Merge this pack as docs.  
2. Run Core + `mercy-security` tests; do not `cargo test --workspace`.  
3. Operator picks **one** Cursor prompt from `CURSOR-AGENT-PROMPTS-2026-09-19.md`.  
4. Counsel cover remains unsigned. ISO application remains HOLD.

Thunder locked. yoi ⚡
