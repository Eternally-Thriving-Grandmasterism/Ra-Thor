# No client secrets in Grok / external models

Date: 2026-08-31  
Audience: a stranger standing in for the operator  
Status: DRAFT procedure. Not legal advice.  
Workspace: **14.15.6**  
Contact: info@Rathor.ai

## When this applies

Any prompt, file, email, or repo paste that contains a client’s confidential matter, credentials, health, children, litigation, or unpublished commercial data.

## Default

**Do not send it.** Local crates, Lattice Chat on-device, and this compliance folder are the working surface. Grok, Claude, X, and other cloud models are optional and external.

## Stranger checklist (stop if any box is unchecked)

1. [ ] The human owner of the matter has **named** the destination (Grok session / Claude / X / other) in writing.
2. [ ] The destination’s retention / no-training term is **known** for this use, or the human accepts the gap in writing.
3. [ ] Secrets are stripped: API keys, tokens, passwords, account numbers, unpublished addresses of family.
4. [ ] The paste is the **minimum** needed for a draft checklist — not the whole file.
5. [ ] Output will be treated as a draft and reviewed by a human before it goes back to a client or a filing.
6. [ ] If the task is legal advice rather than a draft, **stop** and send the human to licensed counsel.

## If you already pasted by mistake

1. Stop further turns that repeat the secret.
2. Tell the operator what destination received it (this session / grok.com / API).
3. Do not invent a “delete from training” guarantee.
4. Rotate any credential that was pasted.

## Code that must not see client matter until gated

- `crates/ai-bridge` `call_claude` — HTTP POST shape to `api.anthropic.com`
- `crates/ai-bridge` `call_grok` — placeholder today; treat as if it could grow a live call
- Browser cards that open `grok.com/share/…` or X
- Any future “production xAI API wrapper” mentioned in `xai-grok-bridge/README.md`

## Allowed without this checklist

Public repo text, this compliance pack, AG-SML license terms, and hypotheticals that contain no real client identifiers.
