# Wrap seats (Cursor inner loop)

**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Outer loop does not write code.** One seat, one PR, then STOP.

Living kit already on `main` after ADOPT-1:

- `docs/ADOPT.md`
- `wrappers/system-prompt.txt`
- `wrappers/custom-instructions/{grok,claude,chatgpt,gemini,cursor}.md`
- `wrappers/local-shim/rathor_wrap.py`
- `skills/ra-thor-employ/SKILL.md`

Do not invent a hosted rathor.ai key proxy. `app/api/grok/route.js` stays research forest.

---

## WRAP-1 — align Lattice Chat constitution — shipped

Merged on `main` as `#511` (`3ab5e5a57`). `js/chat.js` `SYSTEM_PROMPT` and Copy Context quote the employ constitution (workspace 14.15.6). inspect ≠ METR. Independent of xAI.

---

## WRAP-2 — Employ page wrap card — shipped

Merged on `main` as `#512` (`acf40ae98`). `employ.html` lists the four doors from `docs/ADOPT.md`, including Local HTTP wrap. No public rathor.ai key proxy.

---

## WRAP-3PLUS — SSE + models probe + Gemini/Cursor snippets

**Job:** Local shim `stream=true` byte-forwards SSE (`text/event-stream`) after injecting the constitution. Keep JSON forward for `stream=false`. `GET /v1/models` proxies upstream `/models` or returns a one-item stub. Gemini + Cursor paste-box snippets. Docs: curl + Python OpenAI SDK examples against `http://127.0.0.1:8787/v1`.

**FINISH LINE:** `node tests/employ-spine-lock.test.js` asserts `rathor_wrap.py` has no `stream=false only`, contains `text/event-stream` and `/v1/models`, and `gemini.md` / `cursor.md` exist with `14.15.6` and `info@Rathor.ai`.

**Shipped when this PR merges.**

STOP. No WRAP-4.

---

HOLD: family walk, info@Rathor.ai, inspect ≠ METR, independent of xAI, no COEP change, no i18n packs, no Cargo bump, no hosted proxy.

Never claim METR, an xAI product, AgentOS-certified, or RBE as a present fact. Never send email. Do not interview.
