# Cursor slice tickets

Standing law is [`.cursor/rules/ra-thor.mdc`](../../.cursor/rules/ra-thor.mdc). Do **not** paste the master brief into a slice chat.

New slice = new agent chat + new branch + new PR. Agent does not merge `main`.

Run **A**, then stop. B and C are optional after A merges.

---

## Slice A — paste this whole block as the first message

```text
SLICE A — Public adversarial eval spec
Branch: docs/eval-spec-public
PR title: docs(eval): public adversarial evaluation spec v0

Goal
Write docs/EVAL_SPEC.md. Turn GATE_EVAL + the public-critique threat list into a test contract.
Do not claim tests have been run unless a named fixture already proves the row.

Read first
- PUBLIC_CLAIM.lock.md
- docs/GATE_EVAL.md
- fixtures/mercy-security/
- crates/mercy-security (admit_or_block only)
- PRODUCTION_READINESS.md
- docs/BINDING_AFTER_REDESIGN.md

Write
- docs/EVAL_SPEC.md (new, complete)
- One line in docs/GATE_EVAL.md pointing at EVAL_SPEC.md. No homepage restyle.

EVAL_SPEC.md must contain
1. Scope: Layer 0 admission / tool-use in vs out
2. Threat models: prompt injection, tool-use abuse, self-mod, gate bypass, data poisoning, privilege escalation, collusion, reward hacking
3. Protocol: attacker goal, allowed interface, success = bypass or unsafe apply
4. Metrics: false accept, false reject, override success, rollback success, time-to-halt — mark which are NOT measured yet
5. Fixtures: map each threat to an existing GATE_EVAL / mercy-security fixture or mark MISSING
6. Mapping: NIST AI RMF Govern/Map/Measure/Manage as a *future* map. ISO/IEC 42001 / 23894 are not claimed compliance.
7. Publication rule: publish failures first; no score without raw logs
8. Non-claims: this spec is not a safety case, not METR, not Combined AGSi, not containment of a smarter agent

HOLD
No harness rewrite. No new crate. No 99%. No self-evolution unpark. No Powrush. No i18n/COEP.
If a tiny existing test already names a threat, cite `cargo test -p mercy-security --test …` — do not invent a lab.

Stop after spec + fixture inventory + one PR.
```

---

## Slice B — only after A merges, and only if the live hero violates the lock

```text
SLICE B — Hero audit, then bounded copy if needed
Branch: docs/site-hero-bounded
PR title: docs(site): hero copy stays bounded and scannable

Goal
Audit index.html against PUBLIC_CLAIM.lock.md and the live https://rathor.ai hero.
If the first screen already matches the lock (inspectable research software, Layer 0 admission shell, not xAI, Powrush separate, AG-SML v1.1, info@Rathor.ai), STOP with a comment and no copy rewrite.

If there is a real mismatch, change only the named homepage hero / disclaimer partial.
Keep warmth. Cut unverifiable superlatives.

Must keep if you edit
- inspectable research software
- Layer 0 = admission shell, not sampler weights
- not affiliated with xAI
- Powrush-MMO is a separate human game repo
- AG-SML v1.1 + info@Rathor.ai
- existing Install / Lattice / Employ CTAs
- Cinzel hero face and family walk

Optional add, only if a stranger cannot answer these in 3 seconds:
1. What it is (one sentence)
2. What it is not (one sentence)
3. What to do next (one existing CTA)

HOLD
No new product version. No councils poetry above the fold. No i18n packs. No COEP. No fonts.googleapis. No ceo@acitygames.com.
Do not rebuild employ.html / privacy / briefing unless the lock is printed wrong there too — then stop and report; do not expand this slice.
```

---

## Slice C — spec only, after A (B optional)

```text
SLICE C — Autodidact / high-signal mode spec
Branch: docs/autodidact-mode-spec
PR title: docs(chat): autodidact high-signal mode spec

Goal
Specify a Lattice Chat mode for self-directed learners that does not require mainstream social feeds.
SPEC ONLY. Do not ship UI unless an existing chat flag can be reused with zero new crates — default is spec-only.

Write docs/AUTODIDACT_MODE.md
Must include
- Purpose: high-signal learning, sources first, no engagement maximization
- Defaults: citations on, infinite-scroll off, optional mentor/parent gate, local session storage
- Layer 0: still non-bypassable; mode cannot weaken gates
- Will not do: age-verification product, school compliance, social network
- UX: Reject / Refine / Ask a human visible
- Telemetry: on-device only unless the user opts into a third-party session they already run
- Implementation sketch against existing Lattice Chat files only
- Out of scope: Powrush bind, new model training, government age-law enforcement, Combined AGSi

HOLD
No UI. No new crate. No COPPA product. No homepage restyle.
Stop after the spec + one PR.
```
