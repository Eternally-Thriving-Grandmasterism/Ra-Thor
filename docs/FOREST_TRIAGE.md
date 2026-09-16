# Forest triage — FOREST-TRIAGE-1

**Date:** 2026-09-16  
**Seat:** FOREST-TRIAGE-1  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Conductor:** v14 only (`crates/lattice-conductor-v14`)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. Not certified. Not a legal product. Not an AGSi warranty. Combined AGSi stays SURMISE. inspect ≠ METR. Layer 0 = admission shell.

This file maps the on-disk `crates/` research forest against the living **12-member** Tier-1 set. It does **not** absorb the forest, add workspace members, bump versions, rewrite the whitepaper, or ship a self-evolution product.

Lock: [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) · census: [`CRATE_CENSUS.md`](CRATE_CENSUS.md) · tiers: [`TIER_MAP.md`](../TIER_MAP.md) · employ: [`EMPLOY.md`](EMPLOY.md) · wrap: [`ADOPT.md`](ADOPT.md) · cover: [`WHITEPAPER_v4.2.md`](WHITEPAPER_v4.2.md)

Capable · Bounded · Corrigible.

---

## Refresh of CRATE_CENSUS facts

Non-recursive `crates/` git tree at SHA `ef17f7d6d8f15eb868040615b60f896df236b079` (current `main` after fetch): **296 dirs** + **1 stray** `living-valence-organism-tests.rs`.

Recursive walk of **that subtree only** (not the repository root): **267** `Cargo.toml` files (**265** top-level crate manifests, **2** nested under `self-improvement-extensions/experiments`).

Dir count, no-manifest list, and nested-manifest pair match the 2026-09-02 census. The living default set is still **12**. This PR does **not** re-add a path.

Workspace `[workspace.package]` version remains **14.15.6**. Default `members` (quoted from current root `Cargo.toml`; versions not rewritten):

| Path | Package name | Version |
|------|----------------|---------|
| `crates/ra-thor-one-organism` | `ra-thor-one-organism` | **14.15.7** |
| `crates/lattice-conductor-v14` | `lattice-conductor-v14` | **14.15.0** |
| `crates/reality-thriving-transfer` | `reality-thriving-transfer` | **14.18.1** |
| `crates/kardashev-orchestration` | `kardashev-orchestration` | **14.15.0** |
| `crates/github-connector` | `github-connector` | **14.15.0** |
| `crates/gpu-compute-pipeline` | `gpu-compute-pipeline` | **14.15.0** |
| `crates/quantum-swarm` | `quantum-swarm` | **14.15.0** |
| `crates/sovereign-recovery` | `sovereign-recovery` | **14.15.0** |
| `crates/monorepo-intelligence` | `ra-thor-monorepo-intelligence` | **0.3.11** |
| `crates/mercy_tolc_operator_algebra` | `mercy_tolc_operator_algebra` | **0.5.19** |
| `crates/fractal-mercy-ledger-adapter` | `fractal-mercy-ledger-adapter` | `version.workspace = true` (inherits **14.15.6**) |
| `crates/mercy-security` | `mercy-security` | **14.15.5** |

Twelve default members. Version drift is real and is not product-green. This PR does not mass-rewrite crate versions and does not bump `[workspace.package]`.

**On-disk research forest:** 296 crate directories; only 12 default members. **284** directories are not default members (**253** with a top-level `Cargo.toml`, **31** without). Re-add a path to `[workspace].members` to work on one. Do not `cargo test --workspace` and treat it as product-green.

Sister-org clones are out of scope (ADOPT-SISTER-1). Powrush-MMO is a separate human-game repo; do not grow game files in `crates/powrush`.

---

## Re-add to members this PR: none

No candidate was locally added to `members`, tested with `cargo test -p <pkg>`, and then reverted. Default members stay **12**. Named Core Tier-1 tests were not run in this docs-only seat. Do not `cargo test --workspace`.

---

## Never (do not promote / do not productize)

These stay on disk if they already exist. They are not the public offer. Do not add them to default members. Do not treat them as Tier-1.

| path | has top-level Cargo.toml? | recommend | why |
|------|---------------------------|-----------|-----|
| `crates/self-evolution` | no | never | Not a crate and not a product. No `Cargo.toml`. Do not add one. Do not ship a self-evolution product. |
| `crates/lattice-conductor-v13` | yes | never | DEPRECATED. Conductor is **v14 only**. Do not revive as living conductor or Tier-1. |
| `crates/nexi_universal` | yes | never | Never as Tier-1. Broken path deps (`nexi = { path = "../" }`). Stay on disk as Tier-4 lineage. |
| `crates/lattice-conductor` | yes | never | Unversioned conductor; path-deps `crates/self-evolution`. Living member is `lattice-conductor-v14` only. |
| `crates/ra-thor-meta-intelligence` | yes | never | Self-evolution / autonomous-improvement orchestrator language. Combined AGSi stays SURMISE. No self-evolution product. |
| `crates/infinite-evolution-orchestrator` | no | never | No `Cargo.toml`. Self-evolution-adjacent name. Do not invent a product. |
| `crates/self_improvement_orchestrator` | no | never | No `Cargo.toml`. Self-evolution-adjacent name. Do not invent a product. |
| `crates/self-improvement-extensions` | yes | never | Self-improvement utilities; not the public offer. Nested experiments stay forest. |

### Outside `crates/` — never as a public proxy

| path | recommend | why |
|------|-----------|-----|
| `app/api/grok/route.js` | never | Hosted xAI key proxy (`process.env.XAI_API_KEY`). There is **no** public rathor.ai key proxy. Do not productize this route or sister `rathor-grok-proxy`. Operator-held wrap remains [`ADOPT.md`](ADOPT.md). |

---

## Later-inspect (at most three)

Each row is one possible stranger use case the employ / wrap kit (copy-context, skill pack, local HTTP shim, Lattice Chat local backend) does not already cover. Mechanics not proven here are marked unknown. Do not start these seats from this file.

| path | has top-level Cargo.toml? | recommend | why (one sentence) |
|------|---------------------------|-----------|---------------------|
| `crates/mercy-threshold-wasm` | yes | later-inspect | A WASM-portable mercy-threshold bridge would let a stranger embed admit/block in their own page without the Python wrap shim or Lattice Chat local backend; unknown whether the `wasm` feature is more than a stub. |
| `crates/sovereign-shard-genesis` | yes | later-inspect | Take-home shard minting is not a wrap-kit door (copy / skill / local HTTP / Lattice Chat); unknown whether this crate can retarget conductor v14 (it currently path-deps `lattice-conductor-v13`). |
| `crates/rrel-desktop` | no | later-inspect | Unknown whether this Tauri real-estate (RREL) desktop shell is a living employ surface the browser wrap kit does not cover; no top-level `Cargo.toml`. |

Do not add biomimicry / heaven-on-earth / interstellar council directories to members. Family walk stays Home · Chat · Employ · Launch · Moments · Shard · Forge · Contact · Privacy. No COEP change. No i18n packs.

---

## Full table — `crates/` directories that are not default members

Recommend is `leave` unless listed in Never or Later-inspect. `leave` means stay on disk, out of `members`, not this PR.

| path | has top-level Cargo.toml? | recommend |
|------|---------------------------|-----------|
| `crates/abundance-breath` | yes | leave |
| `crates/access` | no | leave |
| `crates/aether_shades` | yes | leave |
| `crates/ai-bridge` | no | leave |
| `crates/ai_bridge` | no | leave |
| `crates/base-reality-anchor` | yes | leave |
| `crates/biological_unifier` | yes | leave |
| `crates/biomimetic` | yes | leave |
| `crates/biomimetic-sovereign-harmony-council` | yes | leave |
| `crates/blueprint-to-production` | yes | leave |
| `crates/bulletproofs_aggregation` | yes | leave |
| `crates/bulletproofs_range` | yes | leave |
| `crates/cache` | no | leave |
| `crates/carbon_credit_oracle` | yes | leave |
| `crates/code_based_crypto` | yes | leave |
| `crates/codex_eternal` | no | leave |
| `crates/common` | no | leave |
| `crates/core-lattice` | no | leave |
| `crates/cosmic-consciousness-expansion-council` | no | leave |
| `crates/council` | yes | leave |
| `crates/deeper_gadgets` | yes | leave |
| `crates/divine_checksum_9` | yes | leave |
| `crates/dynamic-persona-router-council` | yes | leave |
| `crates/enc` | yes | leave |
| `crates/epiphany-bridge` | yes | leave |
| `crates/eternal-nexus-revelation-council` | yes | leave |
| `crates/eternal-sovereign-divine-spark-council` | yes | leave |
| `crates/eternal-sovereign-infinite-horizon-council` | yes | leave |
| `crates/eternal-sovereign-lattice-completion-council` | yes | leave |
| `crates/eternal-sovereign-mercy-lattice-unification-council` | yes | leave |
| `crates/eternal-sovereign-quantum-consciousness-expansion-council` | yes | leave |
| `crates/eternal-sovereign-spark-council` | yes | leave |
| `crates/evolution` | yes | leave |
| `crates/falcon_sign` | yes | leave |
| `crates/fenca` | yes | leave |
| `crates/futarchy_belief_markets` | yes | leave |
| `crates/futarchy_governance` | yes | leave |
| `crates/futarchy_oracle` | yes | leave |
| `crates/futarchy_outcome_prediction` | yes | leave |
| `crates/geometric-intelligence` | yes | leave |
| `crates/grok_arena_pinnacle` | yes | leave |
| `crates/halo2_full_integration` | yes | leave |
| `crates/halo2_gadgets` | yes | leave |
| `crates/halo2_multi_proof` | yes | leave |
| `crates/halo2_proofs` | yes | leave |
| `crates/hash_based_crypto` | yes | leave |
| `crates/hash_based_signatures` | yes | leave |
| `crates/heaven-on-earth-simulator` | no | leave |
| `crates/hotfix_propagator` | yes | leave |
| `crates/hybrid_pqc_threshold` | yes | leave |
| `crates/hyperbolic-neural-network` | yes | leave |
| `crates/hyperbolic-tiling-consciousness` | yes | leave |
| `crates/hyperon-metta-pln` | yes | leave |
| `crates/hyperplonk_recursion` | yes | leave |
| `crates/idea-recycling` | yes | leave |
| `crates/infinite-evolution-orchestrator` | no | never |
| `crates/infinite-horizon-exploration-council` | yes | leave |
| `crates/innovations-generator` | yes | leave |
| `crates/interstellar-operations` | yes | leave |
| `crates/interstellar-sovereign-asset-lattice-council` | no | leave |
| `crates/isogeny_crypto` | yes | leave |
| `crates/kernel` | yes | leave |
| `crates/lasso_recursion` | yes | leave |
| `crates/lattice-conductor` | yes | never |
| `crates/lattice-conductor-v13` | yes | never |
| `crates/lattice_crypto` | yes | leave |
| `crates/legacy_fenca` | yes | leave |
| `crates/legal-lattice` | yes | leave |
| `crates/mercy` | yes | leave |
| `crates/mercy-biojet` | yes | leave |
| `crates/mercy-gel-pterosaur-propulsion-council` | yes | leave |
| `crates/mercy-governance` | yes | leave |
| `crates/mercy-organism` | yes | leave |
| `crates/mercy-propulsion-trait` | no | leave |
| `crates/mercy-quanta-sentinel-council` | yes | leave |
| `crates/mercy-radiation-shield` | yes | leave |
| `crates/mercy-threshold-wasm` | yes | later-inspect |
| `crates/mercy_accumulator` | yes | leave |
| `crates/mercy_antimatter_propulsion` | yes | leave |
| `crates/mercy_ark` | yes | leave |
| `crates/mercy_arkworks` | yes | leave |
| `crates/mercy_axion_propulsion` | yes | leave |
| `crates/mercy_beamed_propulsion` | yes | leave |
| `crates/mercy_bias_mitigation` | yes | leave |
| `crates/mercy_biojet` | yes | leave |
| `crates/mercy_biomimetic_propulsion` | yes | leave |
| `crates/mercy_bls12_381` | yes | leave |
| `crates/mercy_brane_propulsion` | yes | leave |
| `crates/mercy_bulletproofs` | yes | leave |
| `crates/mercy_casimir_propulsion` | yes | leave |
| `crates/mercy_chemical_propulsion` | yes | leave |
| `crates/mercy_circom` | yes | leave |
| `crates/mercy_classic_mceliece` | yes | leave |
| `crates/mercy_commitment` | yes | leave |
| `crates/mercy_cosmological_propulsion` | yes | leave |
| `crates/mercy_curve` | yes | leave |
| `crates/mercy_cygnus_missions` | yes | leave |
| `crates/mercy_dark_energy_propulsion` | yes | leave |
| `crates/mercy_dilithium` | yes | leave |
| `crates/mercy_dragon_crs` | yes | leave |
| `crates/mercy_dream_chaser` | yes | leave |
| `crates/mercy_electric_propulsion` | yes | leave |
| `crates/mercy_ethics_core` | yes | leave |
| `crates/mercy_exotic_propulsion` | yes | leave |
| `crates/mercy_falcon` | yes | leave |
| `crates/mercy_falcon_heavy` | yes | leave |
| `crates/mercy_fflonk` | yes | leave |
| `crates/mercy_field` | yes | leave |
| `crates/mercy_flight_agi` | yes | leave |
| `crates/mercy_fri` | yes | leave |
| `crates/mercy_frodokem` | yes | leave |
| `crates/mercy_fusion_propulsion` | yes | leave |
| `crates/mercy_gadget` | yes | leave |
| `crates/mercy_gating_runtime` | yes | leave |
| `crates/mercy_governance` | yes | leave |
| `crates/mercy_gravitic_propulsion` | yes | leave |
| `crates/mercy_griffin` | yes | leave |
| `crates/mercy_groth16` | yes | leave |
| `crates/mercy_halo2` | yes | leave |
| `crates/mercy_halo2_gadgets` | yes | leave |
| `crates/mercy_hash` | yes | leave |
| `crates/mercy_higgs_propulsion` | yes | leave |
| `crates/mercy_home_fortress` | yes | leave |
| `crates/mercy_hybrid_propulsion` | yes | leave |
| `crates/mercy_hyperplonk` | yes | leave |
| `crates/mercy_ion_propulsion` | yes | leave |
| `crates/mercy_ipa` | yes | leave |
| `crates/mercy_kaluza_klein_propulsion` | yes | leave |
| `crates/mercy_kyber` | yes | leave |
| `crates/mercy_kzg` | yes | leave |
| `crates/mercy_lang_compiler` | yes | leave |
| `crates/mercy_lattice` | yes | leave |
| `crates/mercy_liquid_democracy` | yes | leave |
| `crates/mercy_loop_quantum_propulsion` | yes | leave |
| `crates/mercy_m_theory_propulsion` | yes | leave |
| `crates/mercy_marlin` | yes | leave |
| `crates/mercy_mceliece` | yes | leave |
| `crates/mercy_merkle_tree` | yes | leave |
| `crates/mercy_merlin_engine` | yes | leave |
| `crates/mercy_mev_servicing` | yes | leave |
| `crates/mercy_mimc` | yes | leave |
| `crates/mercy_multiverse_propulsion` | yes | leave |
| `crates/mercy_neutrino_propulsion` | yes | leave |
| `crates/mercy_newhope` | yes | leave |
| `crates/mercy_nova` | yes | leave |
| `crates/mercy_ntru` | yes | leave |
| `crates/mercy_nuclear_propulsion` | yes | leave |
| `crates/mercy_orbital_cleanup` | yes | leave |
| `crates/mercy_orbital_refuel` | yes | leave |
| `crates/mercy_orbital_safety` | yes | leave |
| `crates/mercy_orbital_servicing` | yes | leave |
| `crates/mercy_orchestrator_v2` | yes | leave |
| `crates/mercy_os_aviation` | yes | leave |
| `crates/mercy_os_kernel` | no | leave |
| `crates/mercy_os_principles` | yes | leave |
| `crates/mercy_pairing` | yes | leave |
| `crates/mercy_pedersen` | yes | leave |
| `crates/mercy_picnic` | yes | leave |
| `crates/mercy_plasma_propulsion` | yes | leave |
| `crates/mercy_plonk` | yes | leave |
| `crates/mercy_poly` | yes | leave |
| `crates/mercy_poseidon` | yes | leave |
| `crates/mercy_post_quantum_sig` | yes | leave |
| `crates/mercy_predictive_policing` | yes | leave |
| `crates/mercy_propulsion` | no | leave |
| `crates/mercy_qec` | yes | leave |
| `crates/mercy_quadratic_funding` | yes | leave |
| `crates/mercy_quadratic_voting` | yes | leave |
| `crates/mercy_quanta` | yes | leave |
| `crates/mercy_quantum_gravity_propulsion` | yes | leave |
| `crates/mercy_quantum_propulsion` | yes | leave |
| `crates/mercy_r1cs` | yes | leave |
| `crates/mercy_rainbow` | yes | leave |
| `crates/mercy_raptor_3` | yes | leave |
| `crates/mercy_raptor_3_integration` | yes | leave |
| `crates/mercy_raptor_3_scalability` | yes | leave |
| `crates/mercy_raptor_engine` | yes | leave |
| `crates/mercy_raptor_integration` | yes | leave |
| `crates/mercy_reactionless_propulsion` | yes | leave |
| `crates/mercy_saber` | yes | leave |
| `crates/mercy_shield_deployment` | yes | leave |
| `crates/mercy_shield_law_enforcement` | yes | leave |
| `crates/mercy_sidh` | yes | leave |
| `crates/mercy_signature` | yes | leave |
| `crates/mercy_snark` | yes | leave |
| `crates/mercy_solar_sail_propulsion` | yes | leave |
| `crates/mercy_space_governance` | yes | leave |
| `crates/mercy_spacetime_propulsion` | yes | leave |
| `crates/mercy_spartan` | yes | leave |
| `crates/mercy_sphincs` | yes | leave |
| `crates/mercy_starfish_servicing` | yes | leave |
| `crates/mercy_stark` | yes | leave |
| `crates/mercy_starliner_crew` | yes | leave |
| `crates/mercy_starship` | yes | leave |
| `crates/mercy_starship_fleet` | yes | leave |
| `crates/mercy_steane` | yes | leave |
| `crates/mercy_string_propulsion` | yes | leave |
| `crates/mercy_sumcheck` | yes | leave |
| `crates/mercy_superstring_propulsion` | yes | leave |
| `crates/mercy_t_rex_engines` | yes | leave |
| `crates/mercy_tachyon_propulsion` | yes | leave |
| `crates/mercy_trajectory_agi` | yes | leave |
| `crates/mercy_vacuum_energy_propulsion` | yes | leave |
| `crates/mercy_vdf` | yes | leave |
| `crates/mercy_verkle` | yes | leave |
| `crates/mercy_vulcan_centaur` | yes | leave |
| `crates/mercy_warp_propulsion` | yes | leave |
| `crates/mercy_wormhole_propulsion` | yes | leave |
| `crates/mercy_zero` | yes | leave |
| `crates/mercy_zkp` | yes | leave |
| `crates/mial` | yes | leave |
| `crates/moebius-transformations` | yes | leave |
| `crates/multi-planetary-sovereign-asset-lattice-council` | no | leave |
| `crates/multivariate_crypto` | yes | leave |
| `crates/nexi_universal` | yes | never |
| `crates/nova_folding` | yes | leave |
| `crates/orch-or-biophoton-consciousness` | no | leave |
| `crates/orchestration` | yes | leave |
| `crates/pasta_curves` | yes | leave |
| `crates/patsagi-councils` | yes | leave |
| `crates/patsagi-quantum-valence` | yes | leave |
| `crates/persistence` | no | leave |
| `crates/philotic-web-fusion` | yes | leave |
| `crates/plasticity-engine-v2` | yes | leave |
| `crates/plonk_recursion` | yes | leave |
| `crates/poseidon_hash` | yes | leave |
| `crates/poseidon_merkle` | yes | leave |
| `crates/post-quantum-biomimetic-grokarena-encryption-council` | yes | leave |
| `crates/powrush` | yes | leave |
| `crates/powrush-governance` | no | leave |
| `crates/powrush-mmo-simulator` | yes | leave |
| `crates/powrush_rbe` | no | leave |
| `crates/prometheus_forge` | yes | leave |
| `crates/proof_verifier` | yes | leave |
| `crates/public-sovereign-asset-lattice-core` | no | leave |
| `crates/public_engagement` | no | leave |
| `crates/quantum` | no | leave |
| `crates/quantum-gravity-harmony-council` | yes | leave |
| `crates/quantum-lattice-consciousness-expansion-council` | no | leave |
| `crates/quantum-propulsion-sovereignty-council` | yes | leave |
| `crates/quantum-swarm-orchestrator` | yes | leave |
| `crates/ra-thor-benchmark` | yes | leave |
| `crates/ra-thor-core` | yes | leave |
| `crates/ra-thor-kernel` | yes | leave |
| `crates/ra-thor-mercy-gated-api` | no | leave |
| `crates/ra-thor-meta-intelligence` | yes | never |
| `crates/ra-thor-monorepo-auditor` | yes | leave |
| `crates/ra-thor-post-quantum-sig` | yes | leave |
| `crates/rathor-sovereign-reasoning-engine` | yes | leave |
| `crates/rbe-powrush-bridge` | no | leave |
| `crates/real-estate-lattice` | yes | leave |
| `crates/recursive_snark` | yes | leave |
| `crates/resonance-challenge` | yes | leave |
| `crates/rrel-desktop` | no | later-inspect |
| `crates/sacred-geometry-core` | yes | leave |
| `crates/self-evolution` | no | never |
| `crates/self-improvement-extensions` | yes | never |
| `crates/self_improvement_orchestrator` | no | never |
| `crates/sentinel_mirror` | yes | leave |
| `crates/shard-composer` | yes | leave |
| `crates/shared-valence-field` | yes | leave |
| `crates/soft-sovereign-agency` | yes | leave |
| `crates/soulscan_x10` | yes | leave |
| `crates/soulscan_x9` | yes | leave |
| `crates/sovereign-asset-lattice-expansion-council` | no | leave |
| `crates/sovereign-asset-registry-council` | no | leave |
| `crates/sovereign-core` | yes | leave |
| `crates/sovereign-decentralized-propulsion-fleet-council` | yes | leave |
| `crates/sovereign-shard-genesis` | yes | later-inspect |
| `crates/sovereign_core` | no | leave |
| `crates/spacetime-reducer-bridge` | yes | leave |
| `crates/spartan_valence` | yes | leave |
| `crates/supernova_folding` | yes | leave |
| `crates/sustainable_space_propulsion` | yes | leave |
| `crates/swarm_intelligence` | yes | leave |
| `crates/symbiotic-membrane` | yes | leave |
| `crates/threshold_crypto` | yes | leave |
| `crates/tolc8-genesis-gate` | yes | leave |
| `crates/universal-sovereign-abundance-lattice-council` | yes | leave |
| `crates/web-forge` | yes | leave |
| `crates/websiteforge` | yes | leave |
| `crates/whitesmiths_anvil` | yes | leave |
| `crates/xtask` | yes | leave |
| `crates/zk_stark` | yes | leave |

Counts in this table: **284** rows (253 yes / 31 no) · **8** never · **3** later-inspect · **273** leave.

Crate dirs with no top-level `Cargo.toml` (31) are unchanged from the 2026-09-02 census.

---

## HOLD (this seat)

- Family walk unchanged: Home · Chat · Employ · Launch · Moments · Shard · Forge · Contact · Privacy
- Layer 0 = admission shell. inspect ≠ METR. Independent of xAI.
- Combined AGSi stays SURMISE. No self-evolution product.
- No hosted rathor.ai key proxy. Do not productize `app/api/grok/route.js` or sister `rathor-grok-proxy`.
- No COEP change. No i18n packs. No `Cargo.toml` members add in this PR.
- No workspace version bump.
- Powrush-MMO is a separate repo. Do not grow game files in `crates/powrush`.
- Contact **info@Rathor.ai**. Never `ceo@acitygames.com` on new prose.

Merge gate remains Core Tier-1 focused `-p` tests. Full `--workspace` is not product-green.

Thunder locked. yoi ⚡
