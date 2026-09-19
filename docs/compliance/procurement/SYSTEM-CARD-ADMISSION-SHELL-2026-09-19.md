# System card — Layer 0 admission shell (draft filter)

**System ID:** S1-L0-ADMIT  
**Workspace:** 14.15.6  
**Date:** 2026-09-19  
**Contact:** info@Rathor.ai  
**Status:** inspectable research software. Not certified. Not a legal product. Combined AGSi stays SURMISE.

This card covers **one** surface. Do not reuse it for hiring, credit, essential services, infrastructure, law enforcement, or legal advice. A new use case needs a new card.

---

## 1. Intended use

Unattended ingest on apply-class paths that cross `lattice-conductor-v14::MercyGatedApi::handle_request` is scanned by `mercy-security::IngestionScanner::admit_or_block`.

| Admit | Block |
| --- | --- |
| Threat class `None` or `Low` | `Medium`, `High`, `Critical` |
| | Payload `> 4 MiB` → `PayloadTooLarge` |

Purpose: keep obviously dangerous ingest (remote-code markers, credential headers, selected obfuscation, fail-closed collusion/reward-hack tokens) from mapping onto ambient g. Human Act remains required before any filing, sale, or public legal claim.

## 2. Out of scope (do not sell as)

- Sampler-weight safety, METR time-horizon, or “the model cannot do X”
- Sandbox containment or malware detection
- Employment, credit, housing, education admissions, essential-service eligibility
- Infrastructure / ICS / weapons / medical diagnosis / law enforcement
- Legal product, lawyer, certified compliance engine
- Combined AGSi as a demonstrated capability
- Live xAI or Anthropic API as the default product (`call_grok` is a placeholder; `call_claude` is a live-capable shape and is **not an offer**)

## 3. Components

| Piece | Path | Role |
| --- | --- | --- |
| Policy + scanner | `crates/mercy-security` | admit / block |
| Apply-class edge | `crates/lattice-conductor-v14` `handle_request` | Medium+ ingest must not become ambient g |
| Public fixtures | `fixtures/mercy-security/` | labeled corpus |
| Evidence ledger | `docs/GATE_EVAL.md` | observed vs claimed |
| Contract | `docs/EVAL_SPEC.md` | what a row must prove |
| Boundary | `docs/LAYER_0_RUNTIME_BOUNDARY.md` | where the shell is **not** enforced |
| Egress | `docs/compliance/MODEL-MAP-EGRESS-2026-08-31.md` | on-device vs wrap vs third party |

## 4. Data path (buyer language)

| Mode | What leaves the device | Who is the model owner |
| --- | --- | --- |
| Default local crates + family PWA | Nothing from chat content; CDN may fetch presentation assets | No third-party model |
| User opens Grok card | Whatever the user types at grok.com | **xAI** — name this router in any contract |
| User opens X card | Whatever the user types on X | **X / xAI** |
| `ai-bridge::call_claude` if invoked | Prompt JSON to `api.anthropic.com` | **Anthropic** — not an offer; do not call with client matter |
| Email to info@Rathor.ai | Correspondence the user sends | Operator mailbox |

Rule: if data can leave, treat it as leaving until a written no-training / retention term exists for that path.

## 5. Known failure modes (published, not hidden)

From `docs/GATE_EVAL.md` (do not silently delete):

- Keyword false rejects (docs that mention `api_key`; negation + `subprocess`)
- Depth-capped base64; some nested wraps still ADMIT
- No `IngestionThreat` yet for MCP / tool-call envelopes (`GE-GAP-TOOL-USE`)
- Binding after uncontrolled self-redesign stays **OPEN** (`GE-GAP-SELF-MOD`)
- Human override actor / rationale not fixture-complete (`GE-GAP-HUMAN-OVERRIDE`)
- No measured live false-accept rate (`GE-GAP-LIVE-FA`)
- A paste that never hits `handle_request` is ungated
- Compile green ≠ live safety. Inspect ≠ METR.

## 6. Human-in-the-loop

The employ loop ends in **human Act**. Fluency is not permission. Override, if added, must record: actor, timestamp, rationale, previous verdict, new verdict, policy version. Schema: [`DECISION-RECORD-SCHEMA.md`](DECISION-RECORD-SCHEMA.md) (not a SIEM; not EU AI Act logging).

## 7. Rollback / kill switch (pilot)

- Stop invoking `handle_request` / `mercy-admit`.
- Pin crate + policy version; do not hot-swap gates in a pilot without a written change note.
- Do not unpark `crates/self-evolution` to “fix” a gate.

## 8. Evaluation status

| Harness | Status |
| --- | --- |
| `cargo test -p mercy-security` | Living Core test — admission shell vs fixtures |
| `docs/compliance/evals/` 20+20 | Spec landed, RESULTS **UNMEASURED** |
| METR / third-party lab | **Not claimed** |

Zero is an allowed score. Do not invent numbers.
