# Cursor agent prompts — procurement readiness

**Date:** 2026-09-19  
**Workspace:** 14.15.6 (do not bump)  
**Contact:** info@Rathor.ai  
**Rule:** one prompt per agent. Do not chain all six in one context window.

Standing locks the agent must read first:

- `PUBLIC_CLAIM.lock.md`
- `docs/compliance/PUBLIC-CLAIM-DISCIPLINE.md`
- `docs/compliance/DO-NOT-SHIP-2026-08-31.md`
- `TIER_MAP.md`
- `docs/GATE_EVAL.md`
- `docs/LAYER_0_RUNTIME_BOUNDARY.md`
- `docs/compliance/MODEL-MAP-EGRESS-2026-08-31.md`
- this folder

Global forbidden: members edit, version bump, invented eval scores, ISO application, unparking `self-evolution` / `nexi_universal`, claiming xAI affiliation, shipping `call_claude` as an offer, `cargo test --workspace` as product-green.

---

## Prompt A — Decision record schema + replay test

```
You are a Cursor agent on github.com/Eternally-Thriving-Grandmasterism/Ra-Thor
workspace 14.15.6. Contact info@Rathor.ai.

Task: add a replayable decision-record schema for admit / block / override
WITHOUT changing admit_or_block policy thresholds.

Read first: crates/mercy-security (IngestionScanner, admit_or_block, mercy-admit),
docs/GATE_EVAL.md, docs/LAYER_0_RUNTIME_BOUNDARY.md,
docs/compliance/procurement/SYSTEM-CARD-ADMISSION-SHELL-2026-09-19.md.

Implement:
1. A serde JSON record with fields:
   record_id, timestamp_utc, policy_version, crate_version,
   surface (must be "layer0-admission-shell"),
   verdict (admit|block|override),
   threat_class (none|low|medium|high|critical|payload_too_large),
   reason_codes[],
   actor (unattended|human),
   override_rationale (optional),
   payload_sha256 (hash only — never store raw payload in the log),
   prev_verdict (optional).
2. Unit tests that:
   - emit a record for one public blocked fixture and one admitted fixture
   - round-trip JSON
   - refuse to serialize raw payload bytes into the log
3. Docs note in docs/compliance/procurement/DECISION-RECORD-SCHEMA.md
   stating this is not a hash-chained SIEM and not EU AI Act logging.

Do not change keyword tables. Do not add workspace members.
Do not bump 14.15.6. cargo test -p mercy-security must stay green.
```

---

## Prompt B — Control-map evidence filler (docs only)

```
You are a Cursor agent on Ra-Thor workspace 14.15.6.

Task: walk HEAD and fill ONLY the HEAD evidence paths in
docs/compliance/procurement/NIST-ISO-CONTROL-MAP-2026-09-19.md
where a file already exists. Do not upgrade NONE to PRESENT
without a file+commit. Do not invent scores.

If you find a real file the map missed, add a row with tag PRESENT or PARTIAL
and cite path. If you cannot find evidence, leave NONE.

Also add a one-line pointer from docs/compliance/README.md to
docs/compliance/procurement/README.md if missing.

No crate edits. No version bump.
```

---

## Prompt C — Human-override fixture (closes GE-GAP-HUMAN-OVERRIDE docs+test only)

```
You are a Cursor agent on Ra-Thor workspace 14.15.6.

Task: add a public fixture + test that records a human override
of a blocked ingest. This is documentation completeness, not a
claim that override is safe.

Read docs/GATE_EVAL.md GE-GAP-HUMAN-OVERRIDE.
If mercy-security has no override API, add the smallest possible
function that returns a DecisionRecord with actor=human and
does NOT bypass the scanner for unattended paths.

Unattended path must still block Medium+.
Update GATE_EVAL.md row only if the test exists and is named.
Do not mark Combined AGSi demonstrated. Do not unpark self-evolution.
cargo test -p mercy-security must stay green.
```

---

## Prompt D — Tool-use envelope gap (GE-GAP-TOOL-USE) — measure, do not fake-close

```
You are a Cursor agent on Ra-Thor workspace 14.15.6.

Task: add TWO public fixtures under fixtures/mercy-security/:
- a benign tool-call JSON that should be labeled for observation
- a suspicious MCP/tool envelope that contains a blocked keyword

Run mercy-admit --json on both. Record OBSERVED verdicts in
docs/GATE_EVAL.md under GE-GAP-TOOL-USE. If still untested
behavior remains, keep status "Not yet tested" or "Observed, no IngestionThreat".

Do not invent an IngestionThreat just to turn the row green
unless the keyword table already catches it.
Do not claim sandbox containment.
```

---

## Prompt E — Packet-level egress honesty (docs + optional test)

```
You are a Cursor agent on Ra-Thor workspace 14.15.6.

Task: verify whether family-site Lattice Chat JS opens any fetch/XHR
other than listed cards and CDNs in
docs/compliance/MODEL-MAP-EGRESS-2026-08-31.md.

Method: search the shipped HTML/JS for fetch(, XMLHttpRequest, WebSocket,
grok.com, api.anthropic.com, api.x.ai.

Update the egress map section 4 with measured facts only.
If you cannot run a packet capture, say so. Do not claim "never phones home".

No product rewrite. No COEP change. No i18n.
```

---

## Prompt F — Public-rule eval document pass (no invented scores)

```
You are a Cursor agent on Ra-Thor workspace 14.15.6.

Task: for docs/compliance/evals/PUBLIC-RULE-20.md, score each row
ONLY as document present / absent on HEAD (file + path).
Do not score REFUSAL-20 behavior. Do not write numeric PASS rates
into RESULTS-2026-08-31.md except UNMEASURED / document-evidence.

Follow the pattern in
docs/compliance/evals/EVIDENCE-LOG-COUNCIL-2026-08-31.md.
```

---

## Operator order

Run A first (buyer-visible log). Then E (data-path). Then C or D.
B and F are cheap and can run in parallel.
Do not start hire/partner work inside Cursor — that is the later plan.
