# Autodidact / high-signal mode — spec

**Date:** 2026-09-18  
**Seat:** SLICE C ([`cursor-teams/SLICES.md`](cursor-teams/SLICES.md))  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. **Spec only.** Not shipped UI. Not a crate. Not a product SKU.

This file specifies a Lattice Chat **mode** for self-directed learners that does not require mainstream social feeds.

It does **not** claim the mode is live on [`/chat.html`](https://rathor.ai/chat.html). Compile success ≠ live behavior. Combined AGSi stays **SURMISE**. inspect ≠ METR.

Slice B ([`cursor-teams/HERO_AUDIT.md`](cursor-teams/HERO_AUDIT.md)) already found the homepage lock intact. This seat does not rebuild it.

Capable · Bounded · Corrigible.

---

## Purpose

High-signal learning. Sources first. No engagement maximization.

A learner should be able to study from this device without opening a social feed, a recommendation infinite-scroll, or a follower graph.

The loop is the employ loop ([`EMPLOY.md`](EMPLOY.md)): Intend → Pass the gates → Optional model → Review → Act. This mode changes **how the chat surface presents drafts**. It does not change Layer 0.

| Intend | This mode |
|--------|-----------|
| Learn from named sources | Prefer a citation, a local document, or “source not demonstrated.” |
| Stay off social feeds | Do not require X / Twitter, Facebook, or any follower graph to start a session. |
| Avoid engagement traps | Infinite-scroll off. No streak, no like-rank, no “for you” queue. |
| Keep a household gate optional | A mentor or parent on the same device may lock the session store. That is not age-law enforcement. |

Ra-Thor is inspectable research software. Outputs are drafts. A human reviews them before filing, sale, or public claims.

---

## What this file is / is not

| This file is | This file is not |
|--------------|------------------|
| A mode contract for a later Lattice Chat seat | Shipped chrome on `chat.html` |
| Defaults and HOLD doors for that seat | A new crate, Cargo member, or model |
| A map onto **existing** chat files | Permission to edit those files in this seat |
| Honest about what is **not wired** | A COPPA, school, or age-verification product |

---

## Defaults

When a later seat implements this mode, these are the defaults. They persist in **local session storage** on this device only.

| Default | Value | Why |
|---------|-------|-----|
| Citations | **On** | Sources first. A claim without a source is labeled “source not demonstrated.” |
| Infinite-scroll | **Off** | Bounded transcript. Search the current session. No feed pagination. No “load more recommended.” |
| Mentor / parent gate | **Optional, off unless the household turns it on** | Reuse the existing local passphrase lock. Not an ID check. Not a school roster. |
| Session storage | **Local only** | Same device. Same existing store. No Ra-Thor account. |

These defaults cannot be flipped by a council vote, a prompt, or a “more engaging” A/B flag.

### Citations on

Every reply that names a fact, a paper, a URL, or a file must do one of:

1. Show the source (URL, local document name already injected, or a quote the learner can check), or
2. Say **source not demonstrated**.

Do not invent a DOI, a METR number, or a “validated” score. [`EVAL_SPEC.md`](EVAL_SPEC.md) is the admission-test contract. Do not rewrite it from this mode.

The living Fast Responder (`LOCAL_KNOWLEDGE` in `js/chat.js`) is canned prose. It is **not** a cited research engine. A later seat that turns citations on must not pretend those canned lines are a bibliography.

### Infinite-scroll off

Living `chat.html` already uses a bounded `#chat-messages` box (`max-height: 32vh`, overflow scroll of **this session**, not a feed). Keep that shape.

Do not add:

- recommended-next cards
- auto-play next lesson
- social “for you” ranking
- unread-count engagement loops

Session search (`#search-input` in `chat.html`, wired in `js/chat.js`) stays a **filter of the current session**. It is not a discovery feed.

### Optional mentor / parent gate

Optional means a trusted adult **on the same device** may require the existing passphrase unlock before the store is readable.

Reuse the living lock, do not invent a new identity product:

- Store flag: encrypted envelope already written by `enableEncryption()` in `js/chat.js`
- Unlock UI already exists: `#unlock-overlay` in `chat.html`
- Keys already in play: `rathor-lattice-sessions-v2` (store), optional AES-GCM envelope (`encrypted: true`)

This gate is a household convenience. It is **not**:

- government age verification
- school attendance / FERPA / COPPA compliance
- a remote parent dashboard
- a social graph of guardians

If the passphrase is forgotten, the store is unrecoverable. That is already the living warning. Do not add a backdoor.

### Local session storage

Living keys (do not rename them in this seat):

| Key | File | Role today |
|-----|------|------------|
| `rathor-lattice-sessions-v2` | `js/chat.js` (`STORE_KEY`) | Multi-session history on this device |
| `rathor-voice-settings-v1` | `js/chat.js` (`SETTINGS_KEY`) | TTS pitch / rate / volume |
| `rathor-local-backend-v1` | `js/chat.js` (`BACKEND_KEY`) | Operator-chosen localhost endpoint + model name |
| `rathor-theme` | `js/rathor-theme.js` | Dark / light. Not a learning flag. |

`ENCRYPT_FLAG` (`rathor-lattice-encrypted-v1`) is **declared** in `js/chat.js` and is **not written** on the living path. Encryption is detected from the store envelope (`encrypted: true`). A later seat must not treat the unused constant as a second source of truth.

`privacy.html` mentions IndexedDB **and** localStorage. Living Lattice Chat writes **localStorage only**. That mismatch is noted, not closed here.

A later seat may add **one** new localStorage key, for example `rathor-autodidact-mode-v1`, beside the keys above. Shape (not shipped):

```json
{
  "enabled": true,
  "citations": true,
  "infiniteScroll": false,
  "mentorGate": false
}
```

`citations` defaults true. `infiniteScroll` defaults false and must stay false while the mode is on. `mentorGate` defaults false. No remote sync. No cookie. No analytics pixel.

---

## Layer 0 — still non-bypassable

This mode is chrome and local defaults. It is **not** a gate.

| Law | Source |
|-----|--------|
| Layer 0 is an admission **shell**, not sampler weights. | [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md) · [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) |
| Unattended ingest on apply-class is `IngestionScanner::admit_or_block`. | [`EVAL_SPEC.md`](EVAL_SPEC.md) (read, not edited) |
| No council vote, prompt, or mode flag turns **Rejected → Apply**. | [`architecture/LAYER0_AUTHORITY_LOCK.md`](architecture/LAYER0_AUTHORITY_LOCK.md) |
| Binding after a system redesigns Layer 0 stays **OPEN**. | [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md) |

This mode **cannot**:

- weaken `admit_or_block`
- skip `MercyGatedApi::handle_request` on apply-class
- map Medium+ ingest to ambient *g*
- disable TOLC 8
- treat `js/chat.js` `mercyGate()` (a local keyword regex) as Layer 0

Honesty from the boundary card: a paste that never crosses the scanner is **ungated**. Lattice Chat Fast Responder, WebLLM, and a user-owned localhost sampler do **not** call `admit_or_block` today. This mode does not close that gap. It also does not pretend the gap is closed.

Optional-model apply that **should** change lattice state still must enter `wrap_model_output` ([`WRAP_LLM_INTENTION.md`](WRAP_LLM_INTENTION.md), [`OPTIONAL_MODEL.md`](OPTIONAL_MODEL.md)). Skipping that path means Layer 0 did not run.

---

## Will not do

| Product someone might infer | Law |
|-----------------------------|-----|
| Age-verification product | No ID upload. No face scan. No government age-API. Household passphrase only. |
| School compliance | Not FERPA, not an LMS, not a gradebook, not a district roster. |
| Social network | No follows, likes, DMs, public learner profiles, or feed ranking. |
| COPPA product | Not a children’s service. `privacy.html` lists COPPA as a **design-intent** word, not certification. This mode does not become that product. |

Also will not do: accounts, cloud sync we operate, recommendation ads, or a Ra-Thor-hosted model proxy.

---

## UX — Reject / Refine / Ask a human visible

When (and only when) a later seat ships this mode, every draft shows three actions. They are **not** on living `chat.html` today. This seat does not add them.

| Action | Learner meaning | Gate meaning |
|--------|-----------------|--------------|
| **Reject** | Do not keep this draft as accepted. Stay stopped. | A gate **Rejected** stays Rejected. The button cannot Apply it. |
| **Refine** | Ask again under the same constitution. Sources first. | Refine is a new Intend. It re-enters the loop. It does not waive Layer 0. |
| **Ask a human** | Stop the sampler. Hand the draft to a person. | Human override remains possible. Completeness of override logs is **GE-GAP-HUMAN-OVERRIDE** in [`EVAL_SPEC.md`](EVAL_SPEC.md) — **not measured**. Do not invent a rate. |

Copy the learner sees (short sentences first):

- **Reject** — This draft is refused. Nothing is applied.
- **Refine** — Ask again. Gates stay on.
- **Ask a human** — Stop here. A person reviews.

“Ask a human” on a personal device means: the learner, a household mentor if the optional gate is on, or [info@Rathor.ai](mailto:info@Rathor.ai) for an organization pilot. It is **not** a public forum and **not** a social mention graph.

The Grok Demo and X Demo links already on `chat.html` are optional third-party doors. They are **not** the autodidact path. A later seat may de-emphasize them while the mode is on. This seat does not edit `chat.html`.

---

## Telemetry

On-device only, unless the user opts into a third-party session **they already run**.

| Channel | Living fact | Autodidact law |
|---------|-------------|----------------|
| Ra-Thor backend | There is none. `chat.html` copy: “No backend we control.” | Do not add one. |
| Session store | `localStorage` key `rathor-lattice-sessions-v2` | Stays on this device. |
| Voice | Web Speech API + `rathor-voice-settings-v1` | Device only. |
| Local Server | `fetch` to the operator’s endpoint (default `http://localhost:11434/v1`) | Allowed. That is the user’s Ollama / LM Studio / OpenAI-compatible host. |
| WebLLM | In-browser WebGPU download the user starts | Allowed. Not a Ra-Thor analytics channel. |
| Copy Context | Clipboard via `copyContext()` in `js/chat.js` | Allowed **only** as an explicit paste into a model the user already uses (Grok, Claude, Gemini, ChatGPT, local). That session then follows **that** provider’s policy. |
| Social feeds | Existing “Open X Demo” / Grok Demo anchors | Not required. Not a learning feed. Do not add pixels, beacons, or recommenders. |

No `sendBeacon`. No third-party analytics. No silent upload of history to rathor.ai.

Export / import JSON already on the page is a **user gesture**, not telemetry.

---

## Implementation sketch — existing Lattice Chat files only

**Do not edit these files in this seat.** Name them so a later chat seat can reuse them with zero new crates.

| File | Why it is in scope | What a later seat may do | What it must not do |
|------|--------------------|--------------------------|---------------------|
| `chat.html` | Living Lattice Chat surface (family walk tab **Chat**). Bounded `#chat-messages`. Session chrome. Copy Context. Local Server / WebLLM buttons. Unlock modal. | Add mode toggle + Reject / Refine / Ask a human **if** a named follow-up slice allows HTML. Default this seat: **no edit**. | No homepage. No i18n pack. No COEP. No new social feed. Do not retitle workspace as 14.18.x or 15.x. `v14.18.x` is a **surface** string; product identity stays **14.15.6**. |
| `js/chat.js` | Store, Fast Responder, `mercyGate()`, Local Server, WebLLM, Copy Context, `SYSTEM_PROMPT`, document inject, search. | Read/write one new localStorage object. Keep citations default on. Keep infinite-scroll off. Reuse passphrase lock for optional mentor gate. Surface the three actions. | Do not call that regex `admit_or_block`. Do not add a crate. Do not train a model. Do not send history anywhere except the user-chosen localhost or an explicit Copy Context paste. |
| `wrappers/system-prompt.txt` | Same constitution `SYSTEM_PROMPT` quotes in `js/chat.js`. | A later seat may add two autodidact lines (sources first; source-not-demonstrated). | Do not weaken Layer 0, inspect ≠ METR, or “outputs are drafts.” |
| `js/rathor-theme.js` | Theme key only. | Leave alone unless contrast of the three actions needs the existing theme tokens. | No new theme product. |
| `js/family-nav-2026-08-22.js` | Family walk. | Leave alone. | Do not add a tenth tab. Walk stays Home · Chat · Employ · Launch · Moments · Shard · Forge · Contact · Privacy. |
| `js/site-lock-2026-08-22.js` | Public claim chrome. | Leave alone. | Do not rewrite the lock. |
| `js/pwa-install.js` | Optional install prompt. | Leave alone. | Not a school app store. |
| `privacy.html` | Local-store / no-backend claims. | Later seat may fix the IndexedDB wording if Chat still uses localStorage only. | Not this seat. |
| `docs/OPTIONAL_MODEL.md` · `docs/ADOPT.md` · `docs/EMPLOY.md` | Optional-model doors (Copy Context, skill, local HTTP wrap, Local Server). | Point learners at door 1 or 4. | Do not add a public key proxy. |
| `docs/EVAL_SPEC.md` | Admission-test contract. | Read. | **Do not edit.** |
| `index.html` | Homepage. Slice B already audited. | — | **Do not touch.** |

### Living paths a learner can use **today** (no new UI)

These already exist. They are not Autodidact Mode. They are the files the mode would sit on.

1. Open `/chat.html` — Fast Responder, local sessions, optional passphrase.
2. Inject a local `.txt` / `.md` / `.json` / `.csv` (document button).
3. Optional: Local Server → operator’s `localhost` OpenAI-compatible endpoint.
4. Optional: Copy Context → paste into a model the learner already runs ([`ADOPT.md`](ADOPT.md) door 1).
5. Review the draft. Act only after a human accepts.

No mainstream social feed is required for steps 1–3 or 5. Step 4 is opt-in.

### What is **not wired** (do not invent a demo)

| Capability | Living state |
|------------|--------------|
| Autodidact toggle | **Not present.** No `rathor-autodidact-mode-v1` key. |
| Citations on | **Not present** as a flag. Fast Responder has no source list. |
| Reject / Refine / Ask a human chrome | **Not present** on `chat.html`. |
| Mentor / parent gate as a named control | **Not present.** Passphrase encryption exists as a privacy lock, not a guardian product. |
| `admit_or_block` on chat send | **Not present.** Client `mercyGate()` is a keyword regex only. |
| Override-log completeness | **NOT MEASURED** ([`EVAL_SPEC.md`](EVAL_SPEC.md) **GE-GAP-HUMAN-OVERRIDE**). |

---

## Out of scope

| Item | Law |
|------|-----|
| Powrush bind | Powrush-MMO is a **separate** repo. Do not start a game loop from this lattice. |
| New model training | No dataset harvest. No fine-tune crate. No reward-hacking loop. |
| Government age-law enforcement | No age-API, no ID vendor, no “compliant kids product” badge. |
| Combined AGSi | Research identity label. Stays **SURMISE**. |
| METR | Keyword inspect and chat chrome are not a time-horizon lab. [`MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md). |
| Slice B homepage rewrite | [`cursor-teams/HERO_AUDIT.md`](cursor-teams/HERO_AUDIT.md) already stopped. Do not rebuild. |
| `docs/EVAL_SPEC.md` rewrite | Read only. |
| New crate / Cargo members / conductor-v13 | Forbidden. Conductor stays **v14**. |
| i18n packs / COEP | Forbidden. Chrome Translate stays a new-tab URL. |
| Email / publish / merge `main` | Forbidden. Human / PATSAGi merge gate. |

Resource-Based Economy is a design thesis / simulator goal. Do not state it as present fact.

---

## Non-claims

| This spec is not | Law |
|------------------|-----|
| A shipped learning product | No UI in this seat. |
| A safety case | Chat send does not cross `admit_or_block` today. |
| A school or COPPA offering | Household passphrase ≠ compliance. |
| An xAI / Grok product | Optional Grok session remains under operator gates. Independent of xAI. |
| A social alternative that still ranks attention | Infinite-scroll stays off. |

---

## HOLD (this seat)

- No UI. No `chat.html` / `js/chat.js` / `index.html` edit.
- No homepage. No i18n. No COEP.
- No Cargo. No new crate. No `[workspace].members` add.
- No `EVAL_SPEC.md` rewrite.
- No Slice B rebuild.
- No COPPA / age-verification / school product.
- No Powrush bind.
- Contact **info@Rathor.ai**. Never `ceo@acitygames.com` on new prose.

Stop after this spec + one PR. Agent does not merge `main`.

Thunder locked. yoi ⚡
