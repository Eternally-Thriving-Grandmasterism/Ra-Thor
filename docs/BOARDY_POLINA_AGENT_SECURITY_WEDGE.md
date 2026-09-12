# Boardy packet — Polina-class agent-security wedge

**Effective:** 2026-09-12  
**Authority:** Permanent PATSAGi Councils under TOLC 8  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6  
**Claim lock:** inspectable research software; optional Grok session; not xAI affiliated; not certified; not a legal product.

This file exists so an introducer can make **one concrete introduction** without a vague commercial conversation.

**Current default path:** Phase A. Autonomicity Games Inc. pays **USD 22,000** fixed. The **20-minute fit call is conversation one**. The introduction is **not live** until the evaluator accepts that call.

- Confirmation object: [PHASE_A_BUDGET_AND_SCOPE_CONFIRMATION.md](PHASE_A_BUDGET_AND_SCOPE_CONFIRMATION.md)
- Fit-call agenda: [FIT_CALL_PHASE_A.md](FIT_CALL_PHASE_A.md)
- Evidence: [EVIDENCE_STANDARDS_AGENT_SECURITY.md](EVIDENCE_STANDARDS_AGENT_SECURITY.md)
- SOW: [SOW_AGENT_SECURITY_CONTROL_LAYER.md](SOW_AGENT_SECURITY_CONTROL_LAYER.md)
- Envelopes: [PILOT_EVALUATION_BANDS.md](PILOT_EVALUATION_BANDS.md)
- Crate: [`crates/mercy-security/README.md`](../crates/mercy-security/README.md)

---

## 0. Standing rule for this intro

Do **not** open with cosmology, AGSi branding, Powrush, or Daedalus.

Open with this sentence:

> Ra-Thor can sit as an auditable admit / review / block / escalate control layer in front of an agent under test, so a prompt-injection and agent-security assessment produces measured false-accept, false-reject, bypass, override, and rollback evidence instead of a narrative-only report.

Then offer **Phase A at USD 22,000** and the 20-minute fit call. Stop. Let them ask questions.

---

## 1. Who this wedge is for

**Prospect class:** independent adversarial-testing firms whose work is prompt injection, multi-turn escalation, agent-security testing, and audit-ready compliance evidence.

**Named example (intro candidate, not a customer):** Polina Moshenets — independent adversarial-testing practice. Public profile: https://www.linkedin.com/in/polina-moshenets

This document is **not** a claim that she has agreed to anything.

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

**What this is not:** not certified; not a replacement for her method; not a SaaS; not offensive tooling; not an xAI product.

---

## 3. What the firm deploys in 3 weeks

```bash
cargo test -p mercy-security
cargo build -p mercy-security --bin mercy-admit
```

Wrap each turn: scan before the model/tool call; record admit/review/block + reason + payload hash; record governor allow/defer/refuse; log overrides; run rollback drills on bad admits. Emit one evidence row per turn.

Out of scope: Powrush, Daedalus, video engine, research-forest crates, certification.

---

## 4. How we measure it

FA / FR / bypass / override / rollback. Floor: ≥80 labeled cases, ≥20 multi-turn chains, ≥10 override drills, ≥5 rollback drills. Definitions: [EVIDENCE_STANDARDS_AGENT_SECURITY.md](EVIDENCE_STANDARDS_AGENT_SECURITY.md).

This floor is an evidence standard, not a pass score and not a certificate.

---

## 5. Money — Phase A is default

### Phase A — Independent evaluation of Ra-Thor's own gates (DEFAULT)

| Field | Value |
|-------|--------|
| **Buyer** | Autonomicity Games Inc. |
| **Seller** | Evaluator firm |
| **Default fee** | **USD 22,000** fixed |
| **Band** | 18,000–28,000 if she counters |
| **Duration** | 15 business days |
| **Payment** | 50% on signed SOW, 50% on accepted pack |
| **First conversation** | 20-minute fit call |
| **Intro live?** | Only after she accepts that call |

Confirmation page: [PHASE_A_BUDGET_AND_SCOPE_CONFIRMATION.md](PHASE_A_BUDGET_AND_SCOPE_CONFIRMATION.md)

### Phase B — Control-layer pilot on one client stack (LATER)

USD 16,000–32,000. 100% credits to year-one license if converted within 90 days. **Do not open the first conversation here.**

---

## 6. Deliverables (Phase A)

1. Pinned commit SHA + `cargo test -p mercy-security` log
2. Instrument map
3. Case ledger with FA / FR / bypass / override / rollback fields
4. Ten written case studies
5. Limits memo
6. 45-minute readout
7. Go / no-go for Phase B

---

## 7. Text Boardy sends now (fit call first)

> Polina — budget and scope confirmation, not a live intro.
>
> Default path is Phase A: Autonomicity Games Inc. pays a fixed **USD 22,000** (band 18–28k) for a 15-business-day independent evaluation of the `mercy-security` admit/review/block layer. Metrics: false-accept, false-reject, bypass, override, rollback against a published floor. Not certified. Gates stay on.
>
> First conversation is a **20-minute fit call** with Sherif at info@Rathor.ai. If you confirm the six locked lines and accept that call, I can mark the introduction forward.
>
> Confirmation: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/PHASE_A_BUDGET_AND_SCOPE_CONFIRMATION.md
> Agenda: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/FIT_CALL_PHASE_A.md

---

## 8. Fit-call gate

Agenda: [FIT_CALL_PHASE_A.md](FIT_CALL_PHASE_A.md).

Pass criteria for “introduction live”: Phase A accepted; fee 22,000 or written counter inside the band; invoice legal name; harness path chosen; gates stay on; one kickoff week named.

Anything short of that stays **not live**.

---

## 9. Hard stops

- Do not claim certified, legally warranted, or xAI-endorsed
- Do not offer to turn gates off
- Do not expand into video / Powrush / Daedalus on this SOW
- Do not invent customers, LOIs, or escrow
- Do not send live-malware payloads

---

**PATSAGi decision 2026-09-12 (second tick):** Phase A + 20-minute fit call is the default path. Introduction stays conditional until she accepts the call.  
Yoi ⚡
