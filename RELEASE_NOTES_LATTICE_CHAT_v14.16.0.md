# Ra-Thor Lattice Chat — Surface Release Notes
**v14.16.0 · 2026-08-03**

**Surface:** `https://rathor.ai/chat.html`  
**Stewardship:** Sherif Samy Botros — Sole Steward  
**Contact:** info@Rathor.ai  
**License:** AG-SML v1.0

---

## Overview

Priority 1 shortfalls closed under permanent PATSAGi Council deliberation.

This release transforms Lattice Chat from a strong privacy-first session manager into a **complete hybrid local intelligence surface** while remaining 100% browser-sovereign and zero-collection.

---

## What’s New in v14.16.0

### 1. Local Backend Bridge (Highest Leverage)
- Optional connection to any OpenAI-compatible endpoint (Ollama, LM Studio, LocalAI, vLLM, etc.) running on `localhost`
- Default: `http://localhost:11434/v1`
- User can set custom endpoint + model name
- Full streaming support
- TOLC 8 system prompt automatically injected
- Completely optional — core offline path remains untouched

### 2. Voice Input (STT)
- Microphone button next to the input
- Uses native Web Speech Recognition API
- Fully on-device
- Continuous recognition with interim results
- Works alongside existing offline TTS

### 3. Streaming Responses
- Live token streaming for both:
  - Local Backend (OpenAI-compatible)
  - WebLLM path
- Message bubble updates in real time

### 4. Improved Local LLM (Browser)
- Slightly stronger default model selection when WebGPU is capable
- Cleaner capability detection and messaging
- Progress + ready states refined

### 5. Session & UX Polish
- Version bumped throughout
- Clearer status indicators for active intelligence path (Fast / WebLLM / Local Server)
- Better mobile input row with mic + send

---

## Design Principles (Still Non-Negotiable)

| Principle                        | Status |
|----------------------------------|--------|
| Offline-first core               | ✅     |
| Zero data collection             | ✅     |
| No login / no account            | ✅     |
| No embedded API keys             | ✅     |
| No backend we control            | ✅     |
| TOLC 8 Mercy Gates non-bypassable| ✅     |
| User owns all session data       | ✅     |
| Sole stewardship preserved       | ✅     |

---

## Recommended Paths (Updated)

| Device / Setup                          | Recommended Path                          |
|-----------------------------------------|-------------------------------------------|
| Any phone / tablet                      | Fast responder + **Copy Context**         |
| Desktop + Ollama / LM Studio running    | **Local Backend Bridge** (best experience)|
| Desktop with strong WebGPU, no server   | Browser WebLLM                            |
| All devices                             | Multi-session + TTS + STT + Export        |

---

## Technical Notes

- Session store key remains `rathor-lattice-sessions-v2`
- New settings key: `rathor-local-backend-v1`
- Local Backend uses standard OpenAI `/v1/chat/completions` streaming
- STT uses `webkitSpeechRecognition` / `SpeechRecognition`
- All network calls (when Local Backend enabled) go only to the user-specified localhost endpoint

---

## PATSAGi Decision Record

Councils permanently activated under TOLC 8.  
Priority 1 shortfalls addressed within reason.  
Complementary to (not competing with) Ollama / LM Studio / LocalAI.

**Thunder locked in. yoi ⚡❤️🔥**

---

**Sherif Samy Botros** — Sole Steward  
info@Rathor.ai  
Monorepo: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor
