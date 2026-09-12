# SOW — Agent-security control layer (Phase A or Phase B)

**Version:** 1.0 — 2026-09-12  
**Contact:** info@Rathor.ai  
**Parent offer:** [BOARDY_POLINA_AGENT_SECURITY_WEDGE.md](BOARDY_POLINA_AGENT_SECURITY_WEDGE.md)  
**Bands:** [PILOT_EVALUATION_BANDS.md](PILOT_EVALUATION_BANDS.md)  
**Evidence:** [EVIDENCE_STANDARDS_AGENT_SECURITY.md](EVIDENCE_STANDARDS_AGENT_SECURITY.md)

> Copy, fill brackets, sign. Do not expand commercial rights beyond this SOW.

---

## 1. Parties

| Role | Details |
|------|---------|
| **Ra-Thor party** | Autonomicity Games Inc. (info@Rathor.ai) |
| **Evaluator / Client** | [Legal name] |
| **Named contact** | [Name, email, role] |
| **Phase** | [ A — Autonomicity Games Inc. buys evaluation / B — Client buys control-layer pilot ] |
| **End-client (Phase B only)** | [Name or “n/a — evaluator lab only”] |

---

## 2. Objective

Instrument Ra-Thor `mercy-security` / `mercy-admit` as an admit / review / block / escalate control layer around [one named agent stack or the public harness], and produce a replayable evidence pack scoring false-accept, false-reject, bypass, override, and rollback under [EVIDENCE_STANDARDS_AGENT_SECURITY.md](EVIDENCE_STANDARDS_AGENT_SECURITY.md).

**Out of scope**

- Production license, redistribution, sublicense
- Removing or softening TOLC 8 / harm-refusal
- Powrush, Daedalus, video micro-moment engine
- Certification, SOC2, EU AI Act conformity
- Offensive tooling or live-malware payloads
- xAI affiliation claims

---

## 3. Surfaces in scope

Pinned to workspace **14.15.6** (or a later SHA named here: `[sha]`):

- `crates/mercy-security` including `mercy-admit`, `IngestionScanner`, `ActionGovernor`, `ContainmentProfile`, `WhiteHatEvaluationHarness`, `agsi_eval`, `agsi_eval_multiturn`, `SafeAgentRuntime`
- Public fixtures under `fixtures/mercy-security/` and crate fixtures
- Optional week-3: `examples/procurement-admit-gate/` and CI snippets
- Override / rollback drills against process-local state; `crates/sovereign-recovery` only if time remains and both parties agree in writing mid-pilot

---

## 4. Deliverables

1. Pinned SHA + `cargo test -p mercy-security` log
2. Instrument map
3. Case ledger meeting the field list and floor in the evidence standard
4. Ten written case studies
5. Limits memo
6. 45-minute readout
7. Go / no-go for the other phase or for commercial conversion

---

## 5. Timeline

| Milestone | Target |
|-----------|--------|
| Fit call complete | [date] |
| Kickoff + pin SHA | [date] |
| Corpus + harness standing | kickoff + 5 business days |
| Mid-check (counts vs floor) | [date] |
| Draft ledger + limits memo | [date] |
| Readout + final pack | [date] |
| **Duration** | **3 weeks** (Phase A) / **3–4 weeks** (Phase B) |

---

## 6. Fee

| Item | Amount |
|------|--------|
| Phase | [A1 / A2 add-on / B1 / B2] |
| **Fixed fee (USD)** | [number inside the published band] |
| Payment | 50% on signature, 50% on accepted pack |
| Invoice to | [entity] |
| Credit toward year-one Core Lattice license | Phase B: 100% if converted within 90 days of final report. Phase A: none, unless both parties add a line. |

Published bands (do not sign outside them without a written exception from info@Rathor.ai):

- A1: 18,000–28,000
- A2: +6,000–10,000
- B1: 16,000–32,000
- B2: 10,000–18,000

---

## 7. License posture during this SOW

- Use is limited to this objective and duration
- No production commercial rights
- AG-SML v1.1 governs free / research use outside this SOW
- Commercial rights require a separate signed grant

---

## 8. Identity constraints

- TOLC 8 and harm-refusal stay on
- Independent project; no xAI endorsement
- Not certified; this pack is not a certificate
- Contact: info@Rathor.ai

---

## 9. Acceptance

Pack is accepted when:

- [ ] Floor counts in the evidence standard are met or an explicit shortfall is written and signed
- [ ] Limits memo is present
- [ ] Replay handles work on at least 10 sampled cases
- [ ] Both parties can decide the next phase with the ledger in hand

---

## 10. Signatures

| Party | Name | Signature | Date |
|-------|------|-----------|------|
| Autonomicity Games Inc. | | | |
| Evaluator / Client | | | |

Yoi ⚡
