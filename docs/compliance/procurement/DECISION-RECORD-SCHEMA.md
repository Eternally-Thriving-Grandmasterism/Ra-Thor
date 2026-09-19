# Decision record schema — Layer 0 admission shell

**Date:** 2026-09-19  
**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Crate:** `mercy-security` (`DecisionRecord`)  
**Status:** inspectable research software. Not certified. Not a legal product. Combined AGSi stays SURMISE.

This note names the replayable JSON record for **admit / block / override** on the Layer 0 admission shell. It does **not** change `IngestionScanner::admit_or_block` thresholds or keyword tables.

**This is not a hash-chained SIEM.**  
**This is not EU AI Act logging.**  
**This is not ISO/IEC 42001 evidence by itself.**

A buyer can replay “what the unattended gate said about this hash.” That is the whole claim.

---

## What this is

Unattended ingest still follows the living policy on the system card:

| Admit | Block |
| --- | --- |
| Threat class `None` or `Low` | `Medium`, `High`, `Critical` |
| | Payload `> 4 MiB` → `payload_too_large` |

The record is a sidecar. `admit_or_block` stays the gate. The log stores **SHA-256 of the ingest**, never the raw bytes.

Surface string is fixed: `layer0-admission-shell`.

Policy version string is a **label of the existing rule**, not a new rule:

`admit_or_block.none_low.v1`

Card: [`SYSTEM-CARD-ADMISSION-SHELL-2026-09-19.md`](SYSTEM-CARD-ADMISSION-SHELL-2026-09-19.md).  
Boundary: [`../../LAYER_0_RUNTIME_BOUNDARY.md`](../../LAYER_0_RUNTIME_BOUNDARY.md).  
Evidence ledger: [`../../GATE_EVAL.md`](../../GATE_EVAL.md).

---

## Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `record_id` | UUID string | yes | This row |
| `timestamp_utc` | RFC3339 | yes | When the row was written |
| `policy_version` | string | yes | Living None/Low admit label |
| `crate_version` | string | yes | `mercy-security` Cargo package version |
| `surface` | string | yes | Must be `layer0-admission-shell` |
| `verdict` | enum | yes | `admit` \| `block` \| `override` |
| `threat_class` | enum | yes | `none` \| `low` \| `medium` \| `high` \| `critical` \| `payload_too_large` |
| `reason_codes` | string[] | yes | Stable codes (`tier:…`, `threat:…`, `signal:…`). Not the paste. |
| `actor` | enum | yes | `unattended` \| `human` |
| `override_rationale` | string | override only | Why a human overrode. Empty is refused. |
| `payload_sha256` | 64 hex chars | yes | SHA-256 of UTF-8 ingest. **Hash only.** |
| `prev_verdict` | enum | override only | Verdict this row overrides |

Forbidden log keys: `payload`, `raw_payload`, `content`, `body`, `bytes`, `raw`, `ingest`, `text`.  
`DecisionRecord::from_log_json` refuses those keys. `to_log_json` never emits them.

Override, when used, records actor, timestamp, rationale, previous verdict, new verdict, and policy version (system card §6). The constructor does **not** bypass the unattended scanner. Medium+ still blocks on `admit_or_block`.

Human-override completeness as a public GATE_EVAL metric stays **GE-GAP-HUMAN-OVERRIDE** until Prompt C lands a named gap-close. This schema is the log shape, not that close.

---

## Example (shape only)

```json
{
  "record_id": "00000000-0000-4000-8000-000000000001",
  "timestamp_utc": "2026-09-19T00:00:00Z",
  "policy_version": "admit_or_block.none_low.v1",
  "crate_version": "14.15.5",
  "surface": "layer0-admission-shell",
  "verdict": "block",
  "threat_class": "critical",
  "reason_codes": ["tier:critical", "threat:remote_code_loader", "signal:trust_remote_code"],
  "actor": "unattended",
  "payload_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

The hex above is a **placeholder**, not a measured digest. Do not treat it as an eval score.

---

## How to verify

```bash
cargo test -p mercy-security --test decision_record
cargo test -p mercy-security
```

Named locks:

- public admit fixture `fixtures/mercy-security/benign/model_card_clean.md`
- public block fixture `fixtures/mercy-security/blocked/trust_remote_code_loader.txt`
- JSON round-trip
- refuse raw payload in the log

Do not `cargo test --workspace` and treat it as product-green.

---

## What remains unproven

- Not a chain of hashes. A later row does not commit to a previous row’s digest.
- Not a SIEM, WORM store, or retention product.
- Not Article 12 / Annex IV EU AI Act logging.
- Not a measured FA%, override%, or live production rate.
- A paste that never hits `admit_or_block` / `handle_request` is ungated and has no record.
- Compile green ≠ live safety. Inspect ≠ METR.

Capable · Bounded · Corrigible.
