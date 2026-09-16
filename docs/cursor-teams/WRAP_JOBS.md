# Wrap seats (Cursor inner loop)

**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Outer loop does not write code.** One seat, one PR, then STOP.

Living kit already on `main` after ADOPT-1:

- `docs/ADOPT.md`
- `wrappers/system-prompt.txt`
- `wrappers/custom-instructions/{grok,claude,chatgpt}.md`
- `wrappers/local-shim/rathor_wrap.py`
- `skills/ra-thor-employ/SKILL.md`

Do not invent a hosted rathor.ai key proxy. `app/api/grok/route.js` stays research forest.

---

## WRAP-1 — align Lattice Chat constitution

**Job:** `js/chat.js` `SYSTEM_PROMPT` and Copy Context must quote `wrappers/system-prompt.txt` (or the same sentences). Kill AG-SML v1.0 / AGSi-demonstration / “symbolic AGI lattice” product voice in the live prompt. Keep local backend + Copy Context behavior.

**FINISH LINE:** `node tests/employ-spine-lock.test.js` still green; new assert that `js/chat.js` contains `14.15.6` and `inspect` and does not sell `AGSi demonstration`. Playtest: Copy Context paste includes draft + independent of xAI.

**Out of scope:** theme, i18n packs, COEP, Cargo, Powrush, enabling `app/api/grok/route.js`.

**STOP** after one PR. Next seat is WRAP-2 only if WRAP-1 merged.

---

## WRAP-2 — Employ page wrap card

**Job:** On `employ.html` after The loop, one card: four doors from `docs/ADOPT.md` with links. No new page. No theme rewrite.

**FINISH LINE:** lock test sees `docs/ADOPT.md` and `rathor_wrap.py` and the employ card text “Local HTTP wrap”.

**STOP** after one PR.

---

## WRAP-3 — optional streaming later

Only if WRAP-1 and WRAP-2 shipped and a human still wants SSE on the local shim. Not required for adoption.

---

## Cursor paste (WRAP-1)

You are the inner loop. Outer loop already decided.

Repo: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor  
Fetch current `main`. Workspace **14.15.6**.

Read first: `docs/ADOPT.md`, `docs/OPTIONAL_MODEL.md`, `docs/EMPLOY.md`, `PUBLIC_CLAIM.lock.md`, `wrappers/system-prompt.txt`, `js/chat.js`.

Execute **WRAP-1 only**. One branch. One PR titled `fix(chat): WRAP-1 align SYSTEM_PROMPT with employ constitution`.

HOLD: family walk, info@Rathor.ai, inspect ≠ METR, independent of xAI, no COEP change, no i18n packs, no Cargo bump, no hosted proxy.

Never claim METR, an xAI product, or RBE as present fact. Never send email.

STOP when the PR lands and the lock test is green. List WRAP-2 in the PR body as next inner loop only.
