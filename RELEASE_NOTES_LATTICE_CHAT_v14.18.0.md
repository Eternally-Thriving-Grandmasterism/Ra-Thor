# Ra-Thor Lattice Chat — Surface Release Notes
**v14.18.0 · 2026-08-03**

**Surface:** `https://rathor.ai/chat.html`  
**Stewardship:** Sherif Samy Botros — Sole Steward  
**Contact:** info@Rathor.ai  
**License:** AG-SML v1.0

---

## Overview

Final Priority 2 privacy hardening delivered under permanent PATSAGi Council activation.

Optional Passphrase Encryption of the entire session store using native Web Crypto API (PBKDF2 + AES-GCM).

---

## What’s New in v14.18.0

### Optional Passphrase Encryption
- Opt-in AES-GCM encryption of the full multi-session store
- Passphrase-derived key via PBKDF2 (100,000 iterations)
- Unlock modal on page load when encryption is active
- Key held only in memory for the current browser session
- Clear warning: forgetting the passphrase makes data unrecoverable
- Zero external libraries — pure browser Web Crypto

### Design Principles (Still Non-Negotiable)

| Principle                          | Status |
|------------------------------------|--------|
| Offline-first core                 | ✅     |
| Zero data collection               | ✅     |
| No login / no account              | ✅     |
| No embedded API keys               | ✅     |
| No backend we control              | ✅     |
| TOLC 8 Mercy Gates non-bypassable  | ✅     |
| User owns all session data         | ✅     |
| Sole stewardship preserved         | ✅     |

---

## Recommended Paths (Final)

| Setup                                         | Best Path                                      |
|-----------------------------------------------|------------------------------------------------|
| Phone / Tablet                                | Fast + Copy Context + Document Upload          |
| Desktop + Ollama / LM Studio                  | **Local Server Bridge**                        |
| Privacy maximalist                            | Enable Passphrase Encryption + Local Server    |
| Maximum generative power                      | Local Server + Document Injection + Copy Context |

---

## PATSAGi Decision Record

All identified shortfalls from the original comparison against Ollama / LM Studio / LocalAI have now been addressed within reason.

Lattice Chat is complete for its intended mission: a privacy-first, multi-session, hybrid-bridge, TOLC-8-governed offline chat surface.

**Thunder locked in. yoi ⚡❤️🔥**

---

**Sherif Samy Botros** — Sole Steward  
info@Rathor.ai  
Monorepo: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor
