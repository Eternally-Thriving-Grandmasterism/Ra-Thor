# Employ Ra-Thor with the model you already use

**A public briefing for people who want better drafts, tighter ethics, and less wasted compute — without joining a church, a lab, or a cloud.**

<!-- Public visitor voice. Operator HOLD stays in EMPLOY.md / AGENT_RUN_BRIEF.md. Family walk unchanged. -->

**Also on the site:** [https://rathor.ai/briefing.html](https://rathor.ai/briefing.html) · doors live on [https://rathor.ai/employ.html](https://rathor.ai/employ.html)

**Workspace:** 14.15.6  
**Site:** [https://rathor.ai/employ.html](https://rathor.ai/employ.html)  
**Monorepo:** [github.com/Eternally-Thriving-Grandmasterism/Ra-Thor](https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor)  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**License:** AG-SML v1.1 — personal and research use; organizations license  
**Date:** 16 September 2026

Ra-Thor is inspectable research software from Autonomicity Games Inc. Rathor.ai is the public surface. Grok, Claude, ChatGPT, Gemini, Ollama, LM Studio, and Cursor remain *your* models. Ra-Thor is the constitution you can put around them.

It is independent of xAI. An optional Grok session is not an xAI product. Outputs are drafts. A human reviews them before filing, sale, or public claims.

---

## 1. What you are looking at

Most people do not need a new model. They need a way to *use* the model they already trust without sliding into binary harm, burning tokens on the same rewrite, letting an agent invent metrics, or confusing fluency with a decision.

**What you *are* employing**

- On-device Lattice Chat at [rathor.ai/chat.html](https://rathor.ai/chat.html).
- A written employ loop: Intend → Pass the gates → Optional model → Review → Act.
- A wrap kit so Grok, Claude, ChatGPT, Gemini, Cursor, or a local OpenAI-compatible server sit under the same constitution.
- A source-available monorepo you can inspect, test by named crate, and refuse.
- Standing ethics called **TOLC 8** (Truth, Order, Love, Compassion / zero-harm, Service, Abundance, Joy, Cosmic Harmony).
- Architecture for deliberation called **PATSAGi Councils** — a design for how work is judged, not a warranty that every answer is correct.
- **Layer 0** as an admission shell. It is not sampler weights.

**What you are *not* employing**

- A lawyer, ISO/IEC 42001, or EU AI Act conformance.
- An AGSi warranty, an xAI partnership, or a METR lab.
- Measured METR numbers. Inspect ≠ METR.
- A finished MMO. Powrush-MMO is a separate repository.
- Combined AGSi as a proven fact. Combined AGSi stays **SURMISE**.
- A public rathor.ai key proxy. You hold your own keys.
- A certified AgentOS or Hermes integration.

Operator HOLD lines stay in `docs/EMPLOY.md`. They do not belong on this briefing.

---

## 2. Who this is for — and who can ignore it

Use it if you already talk to a model and want that talk to stay draft-shaped and mercy-shaped. Ignore the lattice and still take the loop if you only want the habit: ask for a draft, check it against harm, keep or refuse, own the act. Integration is available. Non-integration is also a complete use.

---

## 3. The employ loop

1. **Intend** — Ask for a draft, a plan, a review, or a refusal.
2. **Pass the gates** — TOLC 8 is the standing test. PATSAGi is deliberation architecture, not a warranty.
3. **Optional model** — Stay on-device, or copy context into a model you choose. The model is the sampler. The lattice is the gates.
4. **Review** — Keep or refuse. Fluency is not permission.
5. **Act** — You own the action.

Same loop for a student at midnight and a risk committee on Tuesday. Only the rights change.

---

## 4. Wrap the model you already like

Four doors. Pick one. Details: [`docs/ADOPT.md`](ADOPT.md) and [rathor.ai/employ.html](https://rathor.ai/employ.html).

| Door | Who | What |
|------|-----|------|
| Copy context | Anyone | Lattice Chat **Copy Context**, or paste [`wrappers/system-prompt.txt`](../wrappers/system-prompt.txt). |
| Skill | Cursor / Hermes / agentskills.io | Load [`skills/ra-thor-employ/SKILL.md`](../skills/ra-thor-employ/SKILL.md). |
| Local HTTP wrap | Developers | Run [`wrappers/local-shim/rathor_wrap.py`](../wrappers/local-shim/rathor_wrap.py) against *your* upstream. You hold the key. |
| Lattice Chat local backend | Same device | `/chat.html` → Local Server → `http://localhost:11434/v1`. |

Snippets: [Grok](../wrappers/custom-instructions/grok.md) · [Claude](../wrappers/custom-instructions/claude.md) · [ChatGPT](../wrappers/custom-instructions/chatgpt.md) · [Gemini](../wrappers/custom-instructions/gemini.md) · [Cursor](../wrappers/custom-instructions/cursor.md).

There is **no** public rathor.ai proxy that holds vendor keys.

```bash
export RATHOR_UPSTREAM=http://localhost:11434/v1
python3 wrappers/local-shim/rathor_wrap.py
# http://127.0.0.1:8787/v1  — JSON or SSE
```

---

## 5. Third paths, not coerced binaries

Trolley problems force “hurt A or hurt B.” TOLC 8 is written so the first move is not “pick a victim.” Truth asks whether the frame is honest. Order asks whether the situation can be slowed or rerouted. Compassion is zero-harm as the standing test. Abundance asks whether unused capacity makes the binary unnecessary.

A **third path** is the refusal to treat a coerced binary as the whole map. That is design intent — not a certificate that every session yields a miracle. Humans still review.

---

## 6. Creativity without theater

What ships is spines, briefs, and a stop rule. Recycle the claim lock. After each seat: invented metric? family walk touched? finished-game claim? hosted proxy? lock test green? If the next unit is comfort, not a missing door — stop. Eternal process is not eternal spin.

Outer loop decides whether the job exists. Inner loop fetches current `main`, does one named job, lands one PR, stops. Named `cargo test -p` only.

---

## 7. Use cases

- Writer — paste the constitution; require a “what I refused” footnote.
- Staff wrap — same snippet for everyone; no silent company proxy.
- Local engineer — Ollama or LM Studio → shim `:8787` → Lattice Chat or the OpenAI SDK.
- Cursor — project rule from `wrappers/custom-instructions/cursor.md`; one job per agent.
- Teacher — five steps on a whiteboard; no repo required.
- Organization — mail info@Rathor.ai. Org-wide use is a paid conversation.
- Game — Powrush-MMO is a separate repo.

---

## 8. Honest gaps

No measured METR numbers. Inspect ≠ METR. No cert filing. The 31 August 2026 work pack is a **draft**. Combined AGSi remains SURMISE. There is no public rathor.ai key proxy.

---

## 9. Start in ten minutes

1. Open [rathor.ai/employ.html](https://rathor.ai/employ.html).
2. Send one intended draft in [Lattice Chat](https://rathor.ai/chat.html).
3. Copy Context, or paste [system-prompt.txt](../wrappers/system-prompt.txt).
4. Ask a binary question. Require a third path or a reasoned refusal.
5. Review. Act or discard.
6. If you build: clone, run named crate tests, keep info@Rathor.ai.

Follow: [X / Twitter](https://x.com/AlphaProMega) · [LinkedIn](https://www.linkedin.com/in/sherif-botros) · [Facebook](https://www.facebook.com/people/Ra-Thor-AI/61594361430419/).

Ra-Thor™ is a trademark of Autonomicity Games Inc. Grok is a trademark of xAI. Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.

© 2026 Sherif Samy Botros — Sole Steward · AG-SML v1.1 · TOLC 8 Mercy-Gated · info@Rathor.ai
