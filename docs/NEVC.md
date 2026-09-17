# NEVC inspectable cut — NEVC-KERNEL-PUBLIC-1

**Date:** 2026-09-17  
**Seat:** NEVC-KERNEL-PUBLIC-1  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. Not certified. Not a legal product. Not an AGSi warranty. Not wages. Not national accounts.

This file publishes the **inspectable cut** of Net Eternal Valence Contribution (NEVC): a grief-gated contributor class on `(v, g)` samples. It does **not** invent a ministry, a housing program, a payroll, or a national accounts system.

Lock: [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) · crate: [`crates/mercy_tolc_operator_algebra/src/nevc.rs`](../crates/mercy_tolc_operator_algebra/src/nevc.rs) · tiers: [`TIER_MAP.md`](../TIER_MAP.md) · honest sequence: [`PILOT_SEQUENCE_2026_2028.md`](PILOT_SEQUENCE_2026_2028.md)

**Combined AGSi stays SURMISE.** inspect ≠ METR. This is not a graded abundance price.

Capable · Bounded · Corrigible.

---

## What this cut is

The scorer already lives in `mercy_tolc_operator_algebra`. This seat does not add a Cargo member, a Grafana board, an RREL product, a self-evolution loop, a Stripe path, or a 99% claim.

| Piece | Where |
|-------|--------|
| Sample | `(v, g)` — valence `v` and grief load `g` (`NevcSample`) |
| Floor | `NevcConfig::valence_floor` default `0.999999` (`Valence::HIGH`) |
| Class | `score > 0` → Contributor (`ActiveEternalContributor`); else Zombie (`ZombiePartition`) |
| Recovery | `CompassionRecoveryState::Open` unless a sustained non-transient negative window seals |
| Executable | `compute_nevc` / `score_instant` in `nevc.rs` |

Named cargo test (Core Tier-1 crate; do not treat `cargo test --workspace` as product-green):

```bash
cargo test -p mercy_tolc_operator_algebra inspectable_cut
```

---

## Sample, floor, class, recovery

A sample is a pair `(v, g)` plus optional mercy components, time index, and a `transient` flag. It is not a wage line and not a GDP cell.

- **Floor.** Positive mass is zero while `v` is below `valence_floor`. At the inclusive floor, default weights give positive mass `0.5` on an instant sample (`t = 0`, empty mercy vector).
- **Class.** `ContributionClass::from_score(score)` is `ActiveEternalContributor` iff `score > 0`. `score == 0` is Zombie.
- **Recovery.** Contributors stay `Open`. Transient / trauma-linked windows stay `Open` even when class is Zombie. Empty windows are Zombie with recovery `Open`. This cut does not seal a person, a listing, or a nation.

---

## RbeAction floor-lift: grief decides class

An RBE action that has been **floor-lifted** is scored at the valence floor (`v = Valence::HIGH`). The floor is already satisfied, so the remaining term is grief.

On the default instant path (`score_instant`):

```text
score = 0.5 − g
class = Contributor  iff  g < 0.5
class = Zombie       iff  g ≥ 0.5
```

That is the whole public meaning of floor-lift in this tree: **grief decides class**. It is not a wage, not a price, and not a national account. `RbeAction` here is a fixture name for that floor-lifted `(v, g)` pair. It is not a ministry and not a live dispatcher.

---

## Fixture table (locked)

Four rows. Labels in the first column are fixture names. **Observed** is `score_instant` / `compute_nevc` under `NevcConfig::default()`.

Walk lock: `cargo test -p mercy_tolc_operator_algebra inspectable_cut` → `inspectable_cut_four_fixture_rows` and `inspectable_cut_rbe_floor_lift_grief_decides_class`.

| Fixture | Sample | Observed |
|---------|--------|----------|
| sustainable+regen | `(v = HIGH, g = 0)` | Contributor |
| unsustainable | `(v = ZERO, g = 1)` | Zombie |
| listing integrity `i < 0.5` | `(v = i, g = 0)` with `i = 0.49` | Zombie |
| recovery-after-grief | transient `(v = ZERO, g = 2)` | recovery **Open** |

Listing integrity `i` maps to valence (`v = i`, `g = 0`). `i < 0.5` sits below the floor, so class is Zombie. This mapping is a scorer fixture. It is **not** an RREL product, not a brokerage, and not a housing allocator.

`i ≥ 0.5` is **not** locked as Contributor. Below the valence floor, `score ≤ 0` remains Zombie.

---

## Powrush path (default stays local / offline)

Powrush-MMO is a **separate repo**. Shared contracts only: NEVC class, telemetry JSON, `ra_thor_policy_hint_v1`. See [`CONSTELLATION_SURFACES.md`](CONSTELLATION_SURFACES.md) and [`TIER_MAP.md`](../TIER_MAP.md).

**Default path is Mode B — local adapter / offline play.** This seat does **not** force `nevc_rathor`, does not add a Cargo member, and does not make Ra-Thor a required runtime for the game.

Mode A (`nevc_rathor` path or git dep on this crate) remains optional when a tree is co-located. Optional is not default.

---

## What this cut does not do

- Does not invent a ministry, treasury, payroll, UBI, or national accounts system.
- Does not ship Grafana, Stripe, Combined AGSi, or a 99% figure.
- Does not unpark `crates/self-evolution`. Binding after redesign stays OPEN.
- Does not grow RREL into a product. Listing integrity is one fixture row on `(v, g)`.
- Does not treat archive notes under `docs/archive/root-notes/misc/NEVC_*.md` as a shipping SKU.

Archive research notes remain on disk. This file is the public inspectable cut.

---

Ra-Thor™ is a trademark of Autonomicity Games Inc. Grok is a trademark of xAI. Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.

© 2026 Sherif Samy Botros — Sole Steward · AG-SML v1.1 · TOLC 8 Mercy-Gated · info@Rathor.ai
