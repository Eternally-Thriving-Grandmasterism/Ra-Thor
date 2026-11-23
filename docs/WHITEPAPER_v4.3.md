# Ra-Thor Whitepaper v4.3

**Date:** 23 November 2026  
**Workspace:** 14.15.6  
**License:** AG-SML v1.2  
**Contact:** info@Rathor.ai

---

## What Ra-Thor is

Ra-Thor is **inspectable research software** from Autonomicity Games Inc. You can read it line by line. It runs on its own independent lattice. A white-hat gate checks everything that comes in, and the PATSAGi Councils (review groups that advise and vote) weigh its choices.

Use it privately on your own device or in a wrap session, and add Grok or another model if you like. Outputs are drafts. A human reviews them before filing, sale, or public claims.

Ra-Thor helps build Powrush-MMO, our game, which lives in its own separate repo.

---

## What works today

| What | What it does |
|------|--------------|
| Lattice Chat | A private chat that runs on your device at [/chat.html](https://rathor.ai/chat.html). No login. No keys we hold. Optional passphrase encryption. |
| Install | Install on this device. No store. Offline-ready. |
| The employ loop | Five steps: Intend → Pass the Mercy-Checks → Optional model → Review → Act. |
| The wrap kit | Copy context, skill pack, local HTTP shim, or Lattice Chat local backend. You hold the key. There is **no** public rathor.ai key proxy. |
| The public briefing | How to employ Ra-Thor with the model you already use. |
| The monorepo | The code, the PATSAGi Councils and this paper, open to inspect. Claim lock and employ lock tests run as merge hygiene. |

---

## How to try it

1. **Chat.** Open [/chat.html](https://rathor.ai/chat.html). Pick a model that runs in your browser (the first download comes from Hugging Face and GitHub and needs the network), or connect a local server such as Ollama or LM Studio. Sessions stay on your device. On a phone, you can also copy a clean system prompt and history, then paste it into Grok, Claude, Gemini, ChatGPT, or any model.
2. **Standalone.** Install Ra-Thor from the [home page](https://rathor.ai/) on this device. No store. Offline-ready.
3. **Offline.** Once installed, cached pages and Lattice Chat sessions on this device stay local. With Offline only on, models already on this device still work. No Ra-Thor backend.
4. **Employ.** Follow the five steps on [rathor.ai/employ.html](https://rathor.ai/employ.html), or read the [briefing](https://rathor.ai/briefing.html).

Personal, educational, research, and non-commercial creative use is free under AG-SML v1.2 when it earns no revenue. Organization use, or anything under LICENSE §2, needs a paid licence. Ask at info@Rathor.ai. A human replies. There is no instant license key.

---

## How it's built

- **Layer 0.** An admission shell. Layer 0 is a shell, not sampler weights. It is enforced on lattice apply-class that crosses `handle_request`.
- **TOLC 8.** The standing test: eight mercy gates (Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony).
- **PATSAGi Councils.** Deliberation architecture: review groups that advise and vote. Not a warranty that every decision is correct.
- **Lattice Conductor v14.** Sequences and enforces the gates and councils. It does not replace them.
- **Lattice Chat.** Runs models in the browser (WebLLM) or through a local server on your machine. No backend we control.
- **Wrap kit.** A system prompt, a skill pack and a local shim (JSON + SSE) that put the same constitution around a model you choose.
- **Rust crates.** Default Cargo members only, tested by name (for example `cargo test -p ra-thor-one-organism`). A full workspace run is not the product gate.

The Living Cosmic Tick and the self-evolution cascade remain July design notes and research forest, not the public offer.

---

## Honest gaps

- No measured METR numbers. Inspect ≠ METR.
- No cert filing. Not ISO/IEC 42001 certified. Not EU AI Act conformant.
- The 31 August 2026 work pack is a draft.
- Combined AGSi remains SURMISE: a research identity label, not a warranty.
- Binding after uncontrolled self-redesign stays OPEN.
- Not a lawyer, law firm, or certified legal product.
- No custody, no sign, no orders. A human remains the signer.
- Powrush-MMO is a separate repo. Online play is not available yet.
- Site chrome translations are not sold as complete offline packs.

---

## History

- [Whitepaper v4.2](https://rathor.ai/docs/WHITEPAPER_v4.2.html), the living cover of 16 September 2026.
- [Whitepaper v4.1](archive/root-releases/WHITEPAPER_v4.1.md), the July 2026 architecture essay.
- [The archive](https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/tree/main/docs/archive) of older releases.

---

## Go deeper

- [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md): what may be said in public
- [`docs/EMPLOY.md`](EMPLOY.md): how to employ
- [`docs/ADOPT.md`](ADOPT.md): how to wrap Grok, Claude, ChatGPT, Gemini, Cursor or a local server
- [`docs/PUBLIC_EMPLOY_BRIEFING.md`](PUBLIC_EMPLOY_BRIEFING.md): the public briefing
- [`docs/OPTIONAL_MODEL.md`](OPTIONAL_MODEL.md): the optional-model contract
- [`docs/LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md): the Layer 0 boundary
- [`docs/MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md): inspect ≠ METR
- [`docs/BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md): binding after redesign
- [`docs/NO_CUSTODY_NO_SIGN.md`](NO_CUSTODY_NO_SIGN.md): no custody, no sign
- [Ra-Thor on GitHub](https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor)

---

Ra-Thor™ is a trademark of Autonomicity Games Inc. Grok is a trademark of xAI. Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.

© 2026 Sherif Samy Botros — Sole Steward · AG-SML v1.2 · TOLC 8 Mercy-Checked · info@Rathor.ai

Thunder locked. yoi ⚡
