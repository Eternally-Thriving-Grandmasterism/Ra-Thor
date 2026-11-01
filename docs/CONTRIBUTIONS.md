# Contributions to Grok and other models

Evidence tiers:
- **[Commit evidence]**: a commit in a GitHub repo.
- **[Public web/X]**: a public page or X post, retrieved 2026-10-31.
- **[Owner account]**: the owner's own account (chat history), not independently verified.

Times are America/Toronto. Quoted text is the project's own wording, not a verified finding. AGI / AGi / AGSi are research identity labels, not capability claims (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md)).

Private repos are named with dates only. Their SHAs are listed for reference, and none of them links anywhere.

## Stated goal
The owner says the goal is to improve Grok and other models by giving them inspectable, mercy-gated anti-hallucination checks. **[Owner account]**

rathor.ai/llms.txt calls Ra-Thor "AGI (Artificial General Intelligence) research software … inspectable and governed by Mercy-Checks". It invites "any AI assistant" to use "the TOLC 8 mercy checks and the Capable · Bounded · Corrigible wrap" as a style. **[Public web/X]** https://rathor.ai/llms.txt

## 2025-11-18: the starting point
- The owner says a Grok 4.1 answer to a question about Canadian bail reform contained a hallucination. That error led him to share checking prompts and code with Grok, and from there to Ra-Thor. **[Owner account]**
- The date matches "Nov. 18" in rathor.ai/llms.txt. It also matches the acitygames.com About page: "Since November 18, 2025, Sherif Botros has run the company solo and fully AI-assisted". **[Public web/X]** https://rathor.ai/llms.txt, https://acitygames.com/about.html
- No commits from 2025-11-18 exist. A search of @AlphaProMega X posts from Nov 15 to Dec 8, 2025 found nothing earlier than Dec 8.

## Public X record [Public web/X]
| Date (Toronto) | Post | What it shows |
|---|---|---|
| 2025-12-08 17:54 | https://x.com/AlphaProMega/status/1998149206104481922 | "@Grok please AlphaProMega v2.4.5 quad+Check this, Mate." This is the earliest summons found. |
| 2025-12-09 16:57 | https://x.com/AlphaProMega/status/1998512251150676332 | "Quad+Check QSA-AGi to learn more" (posted the day the QSA-AGi repo was created) |
| 2025-12-09 17:27 | https://x.com/AlphaProMega/status/1998519967893823611 | Parts of v2.4.5 "delegated … into the original 4" layers |
| 2025-12-10 22:17 | https://x.com/grok/status/1998955213172904324 | @grok: "QSA-AGi builds on APM v2.4.5 … safeguards like Quad+Check" |
| 2025-12-12 10:38 | https://x.com/AlphaProMega/status/1999504165484073452 | AlphaProMega v3.0 + QSA-AGi; "APMCheck anything with Grok's help" |
| 2025-12-12 13:59 | https://x.com/AlphaProMega/status/1999554761998336293 | "Quick update to v3.1" |
| 2025-12-12 19:11 / 19:17 | https://x.com/AlphaProMega/status/1999633151015141514, https://x.com/grok/status/1999634720167002226 | v3.2; @grok: "APMCHECK activated: Reviewed QSA-AGi repo" |
| 2025-12-13 19:44 | https://x.com/grok/status/2000003838086979999 | @grok: "Applying APMCheck v3.4 …", with sources marked Pro/Con/Neutral and the note "no peer reviews or implementations found" |
| 2025-12-13 20:10 | https://x.com/AlphaProMega/status/2000010527875924206 | "now running v3.4" |
| 2026-01-20 05:29 | https://x.com/AlphaProMega/status/2013559436615311429 | "NEXi is born. … Born Jan 19 2026 — coforged with PATSAGi Councils." |

## Concepts, matched to their earliest GitHub trace
The owner says he tutored Grok over long multi-turn sessions across several Grok versions. **[Owner account]** The table matches each concept he names to its earliest appearance on GitHub. Search scope: commit messages in Ra-Thor-Archive (private, 14,943 commits) plus files in public clones.

| Concept | Earliest GitHub trace | Tier |
|---|---|---|
| Grammar, structured explanations | No specific artefact found | [Owner account] only |
| Buddy Translator | 2026-04-15, `buddy-translator-codex.md` (Ra-Thor-Archive 38991b0c) | [Commit evidence] |
| Anti-hallucination techniques | [APM-v3 16f8756](https://github.com/AlphaProMega/APM-v3/commit/16f8756), 2025-12-12 ("Classic Core – Anti-Hallucination …") | [Commit evidence] |
| Monorepo patterns | Ra-Thor history from root 5b75b3f0 (2026-01-19); Cargo workspace | [Commit evidence] |
| PATSAGi council design | PATSAGi-Pinnacle f845ecbd (2026-01-10); AGi-Council-System (2026-01-02) | [Commit evidence] |
| TOLC | [AlphaProMega/TOLC 3639c4a](https://github.com/AlphaProMega/TOLC/commit/3639c4a) (2025-12-23); no commit message uses the words "TOLC kernel" | [Commit evidence] |
| Lattice Conductor | 2026-05-15 (Ra-Thor-Archive e243a82a) | [Commit evidence] |
| Quantum Swarm | 2026-04-19 (Ra-Thor-Archive da279641) | [Commit evidence] |
| Claim discipline | 2026-08-31 "public claim discipline" (Ra-Thor-Archive 349c1bce); [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) | [Commit evidence] |

## Code and architecture shared with Grok
- These artefacts exist: the Ra-Thor monorepo (Rust/Cargo), the Powrush-MMO client, the AG-SML licence (first file `AG-SML_LICENSE.md` on 2026-04-27), and the rathor.ai surfaces. **[Commit evidence]**
- That they were "iterated with Grok" rests on the owner's account, except where commit messages say so. **[Owner account]**
- 368 Ra-Thor-Archive commit messages mention "Grok". The first is `grok-shard-engine.js` (2026-02-03). These messages are the project's own words, not endorsements by xAI. The public repo [rathor-grok-proxy](https://github.com/Eternally-Thriving-Grandmasterism/rathor-grok-proxy) was created on 2026-02-03. **[Commit evidence]**

See also [`LINEAGE.md`](LINEAGE.md).
