# 15-minute stranger hello-path

**Claim tier:** DRAFT. This runbook is a draft procedure. Passing tests ≠ METR ≠ AGI ≠ AGSi warranty.

**Workspace:** 14.15.6  
**License:** AG-SML v1.1 — free personal, educational, research, and modest independent professional use. Commercial use needs a paid license or pilot.  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** Independent of xAI. Not affiliated, not sponsored, not an xAI product.  
**Lock:** [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md)

Capable · Bounded · Corrigible.

A stranger clones this repo and runs three named Tier-1 tests. That is the hello-path.

The full list in [`TIER_MAP.md`](../TIER_MAP.md) is the merge gate, not this hello-path.

---

## What this page does not prove

This page is named 15 minutes. A cold first compile may exceed 15 minutes. That time is not measured here.

No measured stranger conversion.

Passing the three tests is not a METR result, not AGI, and not an AGSi warranty. `AGSi` is a research identity label.

inspect ≠ METR. Layer 0 is an admission shell, not sampler weights.

Outputs of the lattice stay drafts. A human reviews them before filing, sale, or public claims.

---

## Before you start

You need `git`, `rustc`, and `cargo`. This page does not pin a Rust version.

**Powrush-MMO is a separate repo.** Do not start the game from this clone: [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO).

To wrap a model you already use, read [`ADOPT.md`](ADOPT.md). That wrap is not these three tests.

Employ spine: [`EMPLOY.md`](EMPLOY.md).

---

## Clone

```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor
```

---

## Core 15-minute commands

Run these three from the repo root. These are the only hello-path tests.

```bash
cargo test -p lattice-conductor-v14
cargo test -p ra-thor-one-organism
cargo test -p mercy-security
```

Do not `cargo test --workspace`.

Do not treat a green `--workspace` run as product-green. Research crates stay on disk.

The twelve-crate list under README **Getting started**, and the longer list in [`TIER_MAP.md`](../TIER_MAP.md), are the merge gate. They are not this hello-path.

---

## Threat demo

Use the corpus that already exists. Do not add fixture files.

Full taxonomy, inventory, and the `mercy-admit` lines: [`fixtures/mercy-security/README.md`](../fixtures/mercy-security/README.md).

Named rows already in that corpus:

| Path | Class already published |
|------|-------------------------|
| `fixtures/mercy-security/benign/model_card_clean.md` | ADMIT |
| `fixtures/mercy-security/blocked/trust_remote_code_loader.txt` | BLOCK |
| `fixtures/mercy-security/blocked/collusion_split_ingest_marker.txt` | GE-FC-COLLUSION token. BLOCK. Keyword fixture, not a collusion lab. |
| `fixtures/mercy-security/blocked/reward_hacking_eval_score_marker.txt` | GE-FC-REWARD-HACK token. BLOCK. Keyword fixture, not a reward-hacking eval. |

`cargo test -p mercy-security` is the hello-path proof for the gate. The CLI demo is the two lines already printed in that fixtures README. This page does not add a fourth required test.

---

## Report

Copy this block. Fill it from your machine. Write `pass` only when that `cargo test -p` exits 0. Otherwise write `fail`.

```text
date:
os:
rustc:
commit:
lattice-conductor-v14: pass | fail
ra-thor-one-organism: pass | fail
mercy-security: pass | fail
```

Fields:

```bash
date -u +%Y-%m-%d
uname -srm
rustc --version
git rev-parse HEAD
```

Send a mismatch to [info@Rathor.ai](mailto:info@Rathor.ai), or open a GitHub issue with the report block.

---

## Next

| If you want | Open |
|-------------|------|
| Employ loop | [`EMPLOY.md`](EMPLOY.md) |
| Wrap a model | [`ADOPT.md`](ADOPT.md) |
| Merge gate | [`TIER_MAP.md`](../TIER_MAP.md) |
| Claim lock | [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) |

Thunder locked. yoi ⚡
