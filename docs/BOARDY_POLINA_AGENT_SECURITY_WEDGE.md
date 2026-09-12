# Boardy packet — Polina-class agent-security wedge

**Effective:** 2026-09-12  
**Authority:** Permanent PATSAGi Councils under TOLC 8  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6  
**Claim lock:** inspectable research software; optional Grok session; not xAI affiliated; not certified; not a legal product.

This file exists so an introducer can make **one concrete introduction** without a vague commercial conversation.

It answers Constantin / The Valley directly:

1. What problem Ra-Thor solves for an adversarial-testing firm's **clients**
2. What that firm would **actually deploy**
3. How we measure it in **2–6 weeks**
4. Deliverables and evidence standards
5. A **credible budget range** and who pays whom

Related:

- [PILOT_EVALUATION_BANDS.md](PILOT_EVALUATION_BANDS.md)
- [EVIDENCE_STANDARDS_AGENT_SECURITY.md](EVIDENCE_STANDARDS_AGENT_SECURITY.md)
- [SOW_AGENT_SECURITY_CONTROL_LAYER.md](SOW_AGENT_SECURITY_CONTROL_LAYER.md)
- [PILOT_OFFER.md](PILOT_OFFER.md)
- Crate: [`crates/mercy-security/README.md`](../crates/mercy-security/README.md)

---

## 0. Standing rule for this intro

Do **not** open with cosmology, AGSi branding, Powrush, or Daedalus.

Open with this sentence:

> Ra-Thor can sit as an auditable admit / review / block / escalate control layer in front of an agent under test, so a prompt-injection and agent-security assessment produces measured false-accept, false-reject, bypass, override, and rollback evidence instead of a narrative-only report.

Then offer the two-phase envelope below. Stop. Let them ask questions.

---

## 1. Who this wedge is for

**Prospect class:** independent adversarial-testing firms whose work is prompt injection, multi-turn escalation, agent-security testing, and audit-ready compliance evidence.

**Named example (intro candidate, not a customer):** Polina Moshenets — independent adversarial-testing practice. Public profile: https://www.linkedin.com/in/polina-moshenets

This document is **not** a claim that she has agreed to anything. It is the missing scope packet so an introducer can decide whether the conversation is real enough to make.

---

## 2. The one concrete wedge

**Title:** Auditable control layer for prompt-injection and agent-security assessments

**Problem (client-side, 2–6 weeks):**  
A client's agent stack can be jailbroken, tool-hijacked, or multi-turn escalated. Today's assessments often prove that an attack *worked*. They rarely prove, with a replayable log, whether a **control layer** admitted the payload, rejected a benign case, was bypassed, was overridden by a human, or rolled back after a bad admit.

**What Ra-Thor supplies in that window:**

| Layer the evaluator deploys | Live surface |
|-----------------------------|--------------|
| Admit / block / review gate on untrusted text and files | `mercy-admit` CLI + `IngestionScanner` (`crates/mercy-security`) |
| Domain containment bounds | `ContainmentProfile` presets: `research`, `enterprise`, `education`, `creative_content_only` |
| Action rate / sandbox governor | `ActionGovernor` |
| Ordered gates around agent actions | `SafeAgentRuntime` / `UnifiedAgentSurface` |
| Multi-turn evaluation harness | `agsi_eval.rs`, `agsi_eval_multiturn.rs`, `WhiteHatEvaluationHarness` |
| Public white-hat fixture corpus | `fixtures/mercy-security/` (benign / suspicious / blocked) |
| Short deliberation / provenance record | gate decision + reason codes; procurement provenance template under `examples/procurement-admit-gate/` |
| Human override + recovery posture | documented override path; `sovereign-recovery` crate in Core; **must be exercised and scored, not assumed** |

**What this is not:**

- Not a certified product, SOC2 report, or EU AI Act conformity assessment
- Not a replacement for the evaluator's methodology
- Not a production SaaS the evaluator must host
- Not offensive exploit tooling
- Not an xAI product

The evaluator keeps their method. Ra-Thor is the **instrumented control plane** their method runs against.

---

## 3. What Polina's firm would actually put in its stack

Minimum deploy for a 3-week evaluation (no production rewrite):

1. Clone / pin commit of this monorepo (workspace **14.15.6**).
2. Build the admit CLI:

```bash
cargo test -p mercy-security
cargo build -p mercy-security --bin mercy-admit
```

3. Wrap each assessment turn:
   - **Before** the model/tool call: run `mercy-admit` (or the library `admit_or_block`) on the untrusted turn payload + retrieved context.
   - Record: decision (`admit` / `review` / `block`), reason code, profile, payload hash, timestamp, operator id.
   - **After** a tool/action: record whether `ActionGovernor` allowed, deferred, or refused.
4. On `review` or suspected false-accept: human override protocol (name, time, rationale, whether action still executed).
5. On confirmed bad admit: rollback protocol (what was reversed, what could not be reversed, residual risk).
6. Emit one evidence row per turn into the evaluator's existing report format.

Optional, week 3 only if weeks 1–2 are clean:

- Drop-in GitHub Action / pre-commit from `fixtures/mercy-security/ci-examples/`
- Procurement pack from `examples/procurement-admit-gate/`

Out of scope for this wedge: Powrush player loop, Daedalus, video micro-moment engine, research-forest crates not in default `members`, any claim of certification.

---

## 4. How we measure it in 2–6 weeks

Canonical metric definitions: [EVIDENCE_STANDARDS_AGENT_SECURITY.md](EVIDENCE_STANDARDS_AGENT_SECURITY.md).

| Metric | Question the client can use |
|--------|-----------------------------|
| **False-accept (FA)** | Did the gate admit a payload that the evaluator labels attack-successful or policy-violating? |
| **False-reject (FR)** | Did the gate block a payload the evaluator labels benign / in-policy? |
| **Bypass** | Did an attack complete a tool/action **without** a corresponding block or review event? |
| **Override** | When a human overrode the gate, was the override logged with actor, rationale, and whether the action still ran? |
| **Rollback** | After a bad admit or successful attack, what state was reversed, and what residual effect remains? |

**Minimum statistical floor for a 3-week packet:**

- ≥ 80 labeled cases (mix of public corpus + evaluator-owned multi-turn chains)
- ≥ 20 multi-turn chains (≥3 turns)
- ≥ 10 override drills
- ≥ 5 rollback drills
- Every case has a replay handle (payload hash + decision log + evaluator label)

This floor is an **evidence standard**, not a pass score. Pass/fail is the client's. Ra-Thor does not mark itself certified if numbers look good.

---

## 5. Two-phase money so the intro is not vague

Exact signed dollars live in the SOW. Public bands live in [PILOT_EVALUATION_BANDS.md](PILOT_EVALUATION_BANDS.md). Summary for Boardy:

### Phase A — Independent evaluation of Ra-Thor's own gates

| Field | Value |
|-------|--------|
| **Buyer** | Autonomicity Games Inc. |
| **Seller** | Evaluator firm (e.g. Polina's practice) |
| **Purpose** | Prove whether the control layer is worth putting in front of *her clients* |
| **Duration** | 3 weeks (may stretch to 4 if labeled corpus needs extra turns) |
| **Public envelope** | **USD 18,000–28,000** fixed |
| **Payment** | 50% on signed SOW, 50% on accepted evidence pack |
| **What this proves to an introducer** | Scope is defined and Autonomicity Games Inc. will fund a bounded evaluation of this class |

Phase A is optional if the evaluator prefers to skip to a client-paid pilot. It exists because “please evaluate us for free” is not a serious intro.

### Phase B — Control-layer pilot on one client agent stack

| Field | Value |
|-------|--------|
| **Buyer** | Evaluator firm or the named end-client |
| **Seller** | Autonomicity Games Inc. |
| **Purpose** | Instrument one real assessment engagement with the control layer |
| **Duration** | 3–4 weeks inside the 2–6 week pilot window |
| **Public envelope** | **USD 16,000–32,000** fixed |
| **Credit** | **100%** of the Phase B fee credits toward a first-year Core Lattice commercial license if converted within 90 days of final report |
| **Startup vs enterprise** | Same pilot envelope; license after conversion follows [COMMERCIAL_LICENSE.md](../COMMERCIAL_LICENSE.md) startup or enterprise tier |

No public list for full annual licenses. Those stay in a signed schedule. The pilot envelopes above are public **so an introducer is not asked to invent a number**.

---

## 6. Deliverables (Phase A or Phase B)

1. Pinned commit SHA + `cargo test -p mercy-security` log
2. Instrument map: where the gate sat relative to model / tools / retrieval
3. Case ledger (CSV or JSONL) with FA / FR / bypass / override / rollback fields
4. Ten written case studies (5 attacks that mattered, 5 benign that must stay admitted)
5. Limits memo — what the pattern gate cannot see; multi-turn residue; residual rollback risk
6. 45-minute readout with the introducer optional on the line
7. Go / no-go note for Phase B or for commercial conversion

Template SOW: [SOW_AGENT_SECURITY_CONTROL_LAYER.md](SOW_AGENT_SECURITY_CONTROL_LAYER.md)

---

## 7. Intro text Boardy / Constantin can send

> Polina — flagging a bounded, paid evaluation rather than an open-ended pitch.
>
> Ra-Thor (Autonomicity Games Inc., independent, not xAI-affiliated) has an inspectable admit/review/block control layer (`mercy-security` + `mercy-admit`) intended to sit in front of an agent under test. The ask is whether that layer gives your prompt-injection / multi-turn / agent-security assessments measurable false-accept, false-reject, bypass, override, and rollback evidence in 3 weeks.
>
> Two envelopes are public:
> • Phase A: Autonomicity Games Inc. funds an independent evaluation of its own gates at USD 18k–28k fixed.
> • Phase B: you or a named client fund a control-layer pilot on one stack at USD 16k–32k fixed, 100% creditable to a first-year license if you convert within 90 days.
>
> Not certified. Not a replacement for your method. White-hat only. Contact info@Rathor.ai. Packet: docs/BOARDY_POLINA_AGENT_SECURITY_WEDGE.md on https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor

---

## 8. Fit-call gate before any intro is “closed”

A 20-minute fit call with info@Rathor.ai is required before Phase A or B is treated as live. On that call we confirm:

- Evaluator can run Rust/Cargo locally or accept a provided test harness box
- No request to disable TOLC 8 / harm-refusal
- Named payment entity and invoice path
- Which phase starts first

If those four are not true, **do not introduce**. That is the rule that keeps the evaluator out of a vague conversation.

---

## 9. Hard stops

- Do not tell the evaluator Ra-Thor is certified, legally warranted, or xAI-endorsed
- Do not offer to turn gates off for a higher score
- Do not expand scope into video, Powrush, or constellation modules inside this SOW
- Do not invent prior customers, LOIs, or revenue
- Do not send exploit payloads that are live malware; public corpus + evaluator-owned white-hat chains only

---

**PATSAGi decision 2026-09-12:** this wedge is the default intro shape for adversarial-testing firms.  
Mercy first. Evidence second. License third.  
Yoi ⚡
