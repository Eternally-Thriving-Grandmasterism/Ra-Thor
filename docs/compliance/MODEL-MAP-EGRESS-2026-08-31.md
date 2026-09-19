# Model map + egress matrix — workspace 14.15.6

Date: 2026-08-31  
Pin: workspace **14.15.6** | surveyed from HEAD after `c6357d7`  
Prompt E static survey: 2026-09-19 on `dc48c6b2e` (main after #543). See §4.  
Status: DRAFT inventory. Not a privacy policy. Not a certification. Not an xAI affiliation.  
Contact: info@Rathor.ai  
Related: [`NO-CLIENT-SECRETS-2026-08-31.md`](NO-CLIENT-SECRETS-2026-08-31.md)

Rule: if data can leave the device, treat it as leaving until a written no-training / retention term exists for that path.

---

## 1. Local / on-device (default offer)

| Surface | What actually runs | Data that stays |
| --- | --- | --- |
| Family site + PWA | `index.html`, `sw.js`, Lattice Chat, Shard, Forge | Session store in IndexedDB / localStorage per `privacy.html` |
| Offline Lattice Chat | `/chat.html` | On-device only unless the user opens an optional cloud session |
| Tier 1 Rust crates | `ra-thor-one-organism`, `lattice-conductor-v14`, algebra, etc. | Process-local unless the crate opens a network client |
| `xai-grok-bridge` | Root tree, **not** a default workspace member. Default `offline_mode: true`. “LIVE GROK” branch is a `format!` string, not an HTTP client | No network in the surveyed `lib.rs` |
| `ai-bridge::offline_wrap` | Placeholder local string | No network |
| `Offline-mode.md` | **Design note** listing local GGUF models + IndexedDB RAG | Treat as aspiration until those model files and the loader are pinned as shipped artifacts |

---

## 2. Optional / user-initiated egress

| Surface | Destination | What can leave | Retention / no-training |
| --- | --- | --- | --- |
| `index.html` “Ra-Thor + Grok” / “Build with Grok” cards | `grok.com/share/…` | Whatever the user types in that xAI session | **xAI / Grok policy — not documented here** |
| `index.html` “Ra-Thor on X” | `/go-x.html` → x.com | Whatever the user types on X | **X / xAI policy — not documented here** |
| Email | `info@Rathor.ai` | Correspondence the user chooses to send | Operator mailbox; never sell/share per site copy |
| Optional local LLM bridge named in `privacy.html` | User-typed OpenAI-compatible URL (default `http://localhost:11434/v1`) | `js/chat.js` `connectBackend` GET `{endpoint}/models`; `generateWithBackend` POST `{endpoint}/chat/completions` with system prompt + history + user text. Not localhost-locked. Does not target rathor.ai. | User-controlled. Off-device if the typed URL is off-device. |
| `chat.html` WebLLM button | `js/chat.js` `enableLocalLLM` → `import('https://esm.run/@mlc-ai/web-llm')` then `CreateMLCEngine('Llama-3.2-1B-Instruct-q4f16_1-MLC')` | Module bytes from `esm.run` on click. Model-weight hosts after that import were not packet-captured. Later `generateWithLocalLLM` runs in-page against the loaded engine. | **esm.run / MLC policy — not documented here** |
| Family chrome Google Translate | `js/google-translate-optin.js` `googleHref` → `translate.google.com/translate?…&u=https://rathor.ai…` | New-tab navigation only. No widget inject. Page URL leaves; not Lattice Chat bytes. | **Google policy — not documented here** |

---

## 3. Code paths that can leave the device if invoked

| Path | Measured behavior on HEAD | Decision |
| --- | --- | --- |
| `crates/ai-bridge` `call_grok` | Returns wrap of the string `"Grok response placeholder"`. **No HTTP to xAI in this function.** | Do not advertise as a live Grok API |
| `crates/ai-bridge` `call_claude` | `reqwest` `POST https://api.anthropic.com/v1/messages` with prompt JSON. No key wiring visible in this file; still a real egress *shape* | **Do not call with client matter.** Treat as live-capable. Not an offer. |
| `crates/ai-bridge` `http_client: Client` | Constructed on `new()` | Presence of an HTTP client means this crate is not offline-only |
| `xai-grok-bridge` README “Ready for production xAI API wrapper” | README ambition; surveyed `lib.rs` has no `reqwest` | Claim ≠ implementation |
| CDN on family HTML (2026-08-31 row) | Tailwind / Font Awesome / Google Fonts strings | **Stale on the Prompt E survey set.** `privacy.html` and `css/rathor-home-shell.css` already say those CDNs are gone. First-party `/css/rathor-theme.css` + `/fonts/cinzel/`. See §4. |
| `Offline-mode.md` “Periodic push to NEXi repo (with user consent)” | Design note | **Forbidden** until a consent + destination + retention page exists. NEXi is lineage-only |
| GitHub connector | Read/write to this repo when the operator’s token is present | Operator GitHub account, not a hidden Grok training pipe |

---

## 4. Family-site / Lattice Chat — Prompt E static survey (2026-09-19)

Method: search the files `index.html`, `chat.html`, and `privacy.html` actually load (static `<script>` / `<link>` plus scripts those files inject) for `fetch(`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `navigator.sendBeacon`, `grok.com`, `api.anthropic.com`, `api.x.ai`, `api.openai.com`, `wss://`. `go-x.html` is included as the X-card relay those pages open.

**No packet capture was run in this ticket.** This is a source search on `dc48c6b2e`, not a browser network log and not a “never phones home” proof.

Survey set: `index.html`, `chat.html`, `privacy.html`, `go-x.html`, `sw.js`, `js/chat.js`, `js/week-window.js`, `js/family-nav-2026-08-22.js`, `js/site-lock-2026-08-22.js`, `js/google-translate-optin.js`, `js/i18n-chrome.js`, `js/pwa-install.js`, `js/rathor-theme.js`, `js/rathor-feedback.js`, `js/rathor-unify.js`, `js/science-map-lock.js`, `js/watch-footer-lock.js`, `i18n/en.js`, same-origin `/i18n/{lang}.js` loaders, `css/rathor-theme.css` and its first-party `@import`s.

Not in this survey: `employ.html`, Launch / Shard / Forge / Contact bodies, archive JS, crates, or `call_claude`.

### 4.1 Hits

| File | Symbol | Class | What can leave |
| --- | --- | --- | --- |
| `index.html` | `<a href="https://grok.com/share/…">` (two cards) | User-opened card | Whatever the user types in that xAI tab |
| `index.html` | `<a href="/go-x.html">` | User-opened card | Relays to x.com (see `go-x.html`) |
| `index.html` | GitHub / Follow / `mailto:info@Rathor.ai` anchors | User-opened card | Browser navigation or mail the user sends |
| `chat.html` | `<a href="https://grok.com/share/…">` | User-opened card | Same as home Grok card |
| `chat.html` | `<a href="/go-x.html">` | User-opened card | Same X relay |
| `go-x.html` | `TARGET` / `window.open` / last-resort `<a>` | User-opened card | `https://x.com/i/grok/share/1d126738cd7245b08a5d6ee6154b03dd`. No `fetch` / XHR / WebSocket in this file. |
| `privacy.html` | prose `grok.com` / `x.com` | Dead string | Documentation only |
| `js/google-translate-optin.js` | `googleHref` | User-opened card | New tab to `translate.google.com` with `u=https://rathor.ai` + path |
| `js/family-nav-2026-08-22.js` | Follow / GitHub `<a href>` | User-opened card | Navigation the user clicks |
| `js/science-map-lock.js` | GitHub work-card `<a href>` | User-opened card | Injected on science-map pages; hrefs only |
| `js/week-window.js` | `fetch('/js/week-window.json', { credentials: 'same-origin' })` | Same-origin presentation | Date stamp JSON. Not chat bytes. |
| `sw.js` | `fetch` inside `fetch` handler | Same-origin cache | `if (url.origin !== self.location.origin) return`. GET only. Not chat POST. |
| `index.html` / `js/site-lock-2026-08-22.js` / `js/pwa-install.js` | `createElement('script')` / `serviceWorker.register('/sw.js')` | Same-origin presentation | First-party `/i18n/{lang}.js`, `/js/*`. Not a model API. |
| `js/chat.js` | `connectBackend` → `fetch(endpoint + '/models')` | Live client | GET to user-typed URL (default `http://localhost:11434/v1`). No chat body. |
| `js/chat.js` | `generateWithBackend` → `fetch(endpoint + '/chat/completions')` | Live client | POST JSON: `SYSTEM_PROMPT` + recent history + user text. Called from `sendMessage` when `backendEnabled`. Endpoint is not localhost-locked. |
| `js/chat.js` | `enableLocalLLM` → `import('https://esm.run/@mlc-ai/web-llm')` | Live client (undeclared CDN vs 2026-08-31 Tailwind/FA/Fonts row) | On WebLLM click: module fetch to `esm.run`. Then `CreateMLCEngine`. Weight hosts unknown without a packet capture. |
| `js/chat.js` | `generateWithLocalLLM` → `llmEngine.chat.completions.create` | In-page after load | Prompt stays in the engine object. Does not itself contain a URL. |
| `js/chat.js` | `initSpeechRecognition` → `webkitSpeechRecognition` | Unclassified without capture | Mic transcript can enter the send path. Whether the browser vendor sends audio off-device was not measured. |
| `js/chat.js` / `i18n/en.js` | strings `ollama`, `claude`, `chatgpt`, `grok` | Dead string | Knowledge / copy. No caller builds `api.openai.com` / `api.anthropic.com`. |
| `css/rathor-home-shell.css` | `i.fa-solid` `{ display: none }` | Dead presentation | Comment: “CDN gone.” No Font Awesome URL in the survey set. |

Zero hits in the survey set for: `XMLHttpRequest`, `WebSocket`, `EventSource`, `navigator.sendBeacon`, `wss://`, `api.anthropic.com`, `api.x.ai`, `api.openai.com`.

### 4.2 Still unmeasured

- Packet-level proof of every host a live browser actually contacts (including MLC weight mirrors and Chrome speech)
- No-training / retention terms from xAI, X, Google, esm.run / MLC, or Anthropic as applied to Ra-Thor sessions
- Whether `crates/ai-bridge` `call_claude` is reachable from any binary the site ships
- Retention of `info@Rathor.ai` mail beyond “operator mailbox”

Do not invent those measurements. Do not read this section as “never phones home.”

---

## 5. Public-copy watch (not this file’s rewrite)

`privacy.html` says the site is “designed to be compatible with” GDPR, CCPA, COPPA, and the **EU AI Act**. That is compatibility *language*, not a conformity assessment. Keep it off any cert pack until counsel redlines it.

---

## 6. Operator rule

Default: local.  
Grok: optional session the human opens.  
Never paste client secrets into Grok, Claude, X, or this chat unless the path in [`NO-CLIENT-SECRETS-2026-08-31.md`](NO-CLIENT-SECRETS-2026-08-31.md) is checked.
