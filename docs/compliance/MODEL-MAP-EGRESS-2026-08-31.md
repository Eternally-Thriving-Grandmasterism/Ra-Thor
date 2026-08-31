# Model map + egress matrix — workspace 14.15.6

Date: 2026-08-31  
Pin: workspace **14.15.6** | surveyed from HEAD after `c6357d7`  
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
| Optional local LLM bridge named in `privacy.html` | User’s Ollama / LM Studio / WebLLM | Stays on the user machine if they keep it local | User-controlled |

---

## 3. Code paths that can leave the device if invoked

| Path | Measured behavior on HEAD | Decision |
| --- | --- | --- |
| `crates/ai-bridge` `call_grok` | Returns wrap of the string `"Grok response placeholder"`. **No HTTP to xAI in this function.** | Do not advertise as a live Grok API |
| `crates/ai-bridge` `call_claude` | `reqwest` `POST https://api.anthropic.com/v1/messages` with prompt JSON. No key wiring visible in this file; still a real egress *shape* | **Do not call with client matter.** Treat as live-capable. Not an offer. |
| `crates/ai-bridge` `http_client: Client` | Constructed on `new()` | Presence of an HTTP client means this crate is not offline-only |
| `xai-grok-bridge` README “Ready for production xAI API wrapper” | README ambition; surveyed `lib.rs` has no `reqwest` | Claim ≠ implementation |
| CDN on family HTML | Tailwind, Font Awesome, Google Fonts | Presentation fetch. Not chat content. Still third-party |
| `Offline-mode.md` “Periodic push to NEXi repo (with user consent)” | Design note | **Forbidden** until a consent + destination + retention page exists. NEXi is lineage-only |
| GitHub connector | Read/write to this repo when the operator’s token is present | Operator GitHub account, not a hidden Grok training pipe |

---

## 4. What is *not* measured yet

- No-training terms from xAI, X, Anthropic, or CDNs as applied to Ra-Thor sessions
- Whether `call_claude` is reachable from any binary the site actually ships
- Packet-level proof that Lattice Chat never calls network except listed cards
- Retention of `info@Rathor.ai` mail beyond “operator mailbox”

Do not invent those measurements.

---

## 5. Public-copy watch (not this file’s rewrite)

`privacy.html` says the site is “designed to be compatible with” GDPR, CCPA, COPPA, and the **EU AI Act**. That is compatibility *language*, not a conformity assessment. Keep it off any cert pack until counsel redlines it.

---

## 6. Operator rule

Default: local.  
Grok: optional session the human opens.  
Never paste client secrets into Grok, Claude, X, or this chat unless the path in [`NO-CLIENT-SECRETS-2026-08-31.md`](NO-CLIENT-SECRETS-2026-08-31.md) is checked.
