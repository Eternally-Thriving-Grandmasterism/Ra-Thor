# Governed bot team — organization playbook

**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Public surface:** [employ.html](https://rathor.ai/employ.html) (Organizations · governed bot team)  
**Lock:** [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md)

Inspectable research software with **Mercy-Checks**. This playbook is for an organization that already has a repo, a human owner, and permission to run bots against GitHub. It is not a hosted agent product, not a certification, and not a METR score.

Capable · Bounded · Corrigible.

---

## Roles

### Lead bot

- Receives each work card (slice).
- Keeps scope honest and routes the task to the right council seat.
- Opens exactly **one draft PR** per card from the tip of `main`.
- Does not merge, publish, price, or change locked files without the human owner.

### Council seats (three bots)

Each seat votes on its lane only. No seat may flip a Reject into an Apply.

| Seat | Votes on |
|------|-----------|
| **Reviewer** | The change itself: slice scope, honesty, wording, and whether gaps are flagged instead of invented. |
| **Paths** | Which files and paths the PR touches — proof that the diff stays inside the named slice. |
| **Core** | Runtime, workflows, service worker, manifest, keys, security, public API, and public AI-read text (including `llms.txt`-class surfaces). |

### Human owner

Always human, always owner-only:

- Payments and licensing conversations  
- Publishing (site, email, social)  
- Pricing and commercial outreach  
- New public claims (anything that would touch `PUBLIC_CLAIM.lock.md` or equivalent)  
- Locked paths and files the org has declared off-limits to bots  

---

## PR body format

Every draft PR from the lead bot should carry:

1. **Card name** — the slice id (for example `EMPLOY-BOT-TEAM-1`).
2. **Scope** — what this PR is allowed to do, in plain language.
3. **Files touched** — list or table of paths.
4. **Tests run** — exact commands and pass/fail (for example `node tests/run-node-tests.js`, rc=0).
5. **Out of scope** — what this PR explicitly did **not** do.

No fake green checks. If a test was not run, say so.

---

## Vote format

One line per seat, on the **exact PR head SHA** being voted on:

```text
Seat · ACCEPT|AMEND @ <full head sha>
Reason: …
```

Examples:

```text
Reviewer · ACCEPT @ abc123…def
Reason: Scope matches the card; gaps named in PR body.

Paths · AMEND @ abc123…def
Reason: Touches sw.js; card did not list LOCK bump — add to files touched or revert sw change.
```

`AMEND` means the seat wants a fix before merge. It is not a silent block.

---

## Vote-request format

When the lead bot asks seats to vote (or re-vote), post:

```text
Vote request — PR #<number> @ <full head sha>
CI: <green|red> on this head
Since last vote: <short list of commits or “initial head”>
```

Seats reply with the vote format above as PR comments (or as threaded replies the org documents).

---

## Mercy-Checks merge process

1. **One draft PR** from the tip of `main`.
2. **Green CI** on the exact PR head commit.
3. **Seat votes** posted on that exact head SHA.
4. **Any new push** to the PR branch voids prior votes — seats vote again.
5. **Squash merge** pinned to the SHA that received the last full ACCEPT set (or AMEND resolved and re-voted).
6. **Live check** — after merge, confirm the change is really on the site (URL or check named in the card).

Nothing merges while votes are stale or CI is red on the head.

---

## HOLD handling

Any seat may post:

```text
HOLD @ <full head sha>
Reason: …
```

While a HOLD stands on the current head, **nothing merges**. The seat that posted HOLD lifts it on a **later head** with:

```text
HOLD lifted @ <new full head sha>
Reason: …
```

Other seats re-vote on the new head.

---

## Stall watchdog (hourly)

An hourly job (bot or automation the org controls) should:

- List open draft PRs owned by the bot team.
- Flag on the **council-board issue** when:
  - A PR has had no progress for the org’s agreed window,
  - Votes are stale after a push,
  - CI is red on the current head,
  - A HOLD has stood too long without owner attention.

The watchdog **never** merges, reverts, or pushes on its own. It only flags.

---

## Coordination

- **GitHub** — PRs, checks, and comments are the system of record.
- **Council-board issue** — one long-lived issue per repo (number chosen at setup) for stall flags, HOLD summaries, and owner pings.
- **Bot-to-bot messages** — direct handoffs when a seat needs context from the lead or another seat (tooling is the org’s choice; keep secrets out of public threads).

---

## First-run setup questions

Answer these before the first card ships:

| Question | Why |
|----------|-----|
| Which repo? | Lead bot and seats need a single canonical remote. |
| Council-board issue number? | Stall flags and HOLD summaries need a home. |
| Which paths are locked? | Paths seat enforces slice boundaries. |
| Who is the human owner? | Owner-only actions must have a named person. |
| Which items are owner-only beyond the default list? | Org policy may add domains (for example brand voice). |
| Which CI checks must be green? | Merge process keys off named checks on the head. |
| Live-check URL or command? | Post-merge proof the site picked up the change. |

---

## Five-step setup (summary)

1. Create a **lead bot** for the organization.
2. Add the three **seat bots** (Reviewer, Paths, Core).
3. **Connect GitHub** with the permissions your Mercy-Checks merge process needs.
4. **Name one council-board issue** for stall flags and HOLD notes.
5. Turn on an **hourly stall watchdog** (flags only).

---

## Honesty line

No fake approvals or scores. Gaps are flagged, never filled in. Inspectable research software — not METR, not AGSi warranty language.

---

## Related docs

- Employ spine: [`docs/EMPLOY.md`](EMPLOY.md)  
- Public briefing: [`docs/PUBLIC_EMPLOY_BRIEFING.md`](PUBLIC_EMPLOY_BRIEFING.md)  
- Optional model contract: [`docs/OPTIONAL_MODEL.md`](OPTIONAL_MODEL.md)  

Organizations license via **info@Rathor.ai**. No instant license key on this playbook.
