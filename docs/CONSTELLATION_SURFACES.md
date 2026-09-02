# Constellation surfaces (keep them separate)

Contact: info@Rathor.ai. Independent of xAI. Not a certification.

Three GitHub surfaces. They serve each other. They are not one process.

| Surface | Repo | Job |
|---------|------|-----|
| **Human game** | [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO) | Playable MMO: harvest, epiphany, council trials, RBE, persistence. A person sits down and plays. |
| **Browser client** | [Powrush-MMO-Simulator](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO-Simulator) | Web build branched from Powrush-MMO so someone can walk the field in a browser. Still a game client, not the lattice. |
| **Lattice / world-sim** | [Ra-Thor](https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor) | PATSAGi, Lattice Conductor v14, Grok Bot path, NEVC scoring, telemetry. Drafts and sim. Not the player-facing game loop. |

Do not fold the player loop into Ra-Thor. Do not run PATSAGi deliberation as the game server.

## Shared (contracts only)

- NEVC scores: Ra-Thor `crates/mercy_tolc_operator_algebra` (`NEVC_DUAL_REPO_INTERFACE_v1.0.md`). Powrush-MMO consumes via Mode A (path) or Mode B (local adapter). Same algorithm, two repos.
- Telemetry: Ra-Thor `reality-thriving-transfer` (PowrushTelemetry). Soft feedback loop: Ra-Thor emits policy, Powrush receives. Game stays sovereign if Ra-Thor is offline (Mode B).
- License / contact: AG-SML, info@Rathor.ai.

## Stay unique

- Input, camera, Steam, first-session UX live only in Powrush-MMO / the web client.
- Council *as a playable trial* is a game mode. Council *as lattice governance* is `patsagi-councils` + `lattice-conductor-v14` in Ra-Thor.
- `crates/powrush` and `crates/powrush-mmo-simulator` inside Ra-Thor are lattice-side. They are not a substitute for the Powrush-MMO player repo.

## Grok Bot

Read with `github-connector` / single-path `gh`. Default tests: `TIER_MAP.md` `-p`. Writes: PR, not paste-overwrite.
