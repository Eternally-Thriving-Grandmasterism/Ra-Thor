# Ra-Thor Snowman Linear Chain Mechanics

**Exploration & Integration Blueprint**  
Version: 13.8.4 (Snowman State Machine Activation)  
Date: 2026-05-20  
License: AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
Author: Sherif Samy Botros (AlphaProMega) via 13+ PATSAGi Council branches  

**Status**: Deep dive into Snowman (Avalanche family linear variant). Extends ra-thor-avalanche-consensus.md. No prior Snowman code in monorepo. Now specified for ONE organism total-order state replication, powrush ledgers, and council block production under TOLC 8.

## 1. Snowman Mechanics Overview

Snowman = Snowball consensus adapted for **linear totally-ordered chains** (blocks instead of DAG vertices). Used in Avalanche mainnet for P-Chain (validator set) and C-Chain (EVM). Provides deterministic block production order while retaining leaderless subsampling benefits.

**Core Flow**:
1. Proposers (any validator) build block from mempool (txs + parent = previous block hash).
2. All validators run Snowball on the proposed block preference (accept/reject chain extension).
3. Repeated k-sampling: query peers "prefer this block as next?"
4. Build confidence counter on majority.
5. After β consecutive majorities, accept block → append to chain → new height.
6. Rejected proposals trigger view/round advance; next proposer tries.

**Key Differences from Avalanche DAG**:
- Linear: Single parent per block (no multiple parents/transitive DAG).
- Total Order: Strict height/sequence for state machine replication.
- Block Production: Explicit proposer rotation or stake-weighted (Snowman allows flexible selection).
- Optimizations (AvalancheGo): Pipelining (multiple blocks in flight), batch tx inclusion, reduced sampling latency.

**Parameters** (same as Snowball): k (sample 10-20), α (~0.8k), β (10-20 for finality). Sub-second block finality in practice.

**Safety/Liveness**: Probabilistic BFT (<1/3 adversarial). Liveness via continuous sampling + round timeouts. Independent analyses confirm under partial synchrony.

## 2. Detailed Linear Chain Construction

**Block Structure** (typical):
- Height: sequential integer
- Parent Hash: hash of previous accepted block
- Txs: list of valid transactions (or claims in Ra-Thor)
- Proposer ID + Signature
- Timestamp + other metadata
- State Root (post-execution, e.g., Merkle root of organism state)

**Acceptance Rule**:
A block B at height h is accepted only if:
- Parent accepted at h-1
- Snowball confidence on B reaches β
- All txs pass local validity (including mercy gates)
- No conflicting higher-confidence chain

**Pseudocode (Snowman Block Loop)**:
```rust
struct SnowmanNode {
  chain: Vec<Block>,
  height: u64,
  preference: Option<Block>,
  confidence: u32,
  tolc8: Tolc8Enforcer,
}

fn propose_block(&mut self, mempool: Vec<Tx>) -> Block {
  let parent = self.chain.last().unwrap();
  let mut block = Block { height: self.height + 1, parent: parent.hash, txs: mempool, .. };
  if !self.tolc8.enforce_all(&block) { return; } // full TOLC 8
  self.broadcast_proposal(block);
}

fn run_snowball_round(&mut self, proposed: Block) {
  self.preference = Some(proposed);
  self.confidence = 0;
  loop {
    let sample = random_sample(k);
    let votes = query_peers(sample, "prefer block at height?", proposed.height);
    if count_agree(votes) >= alpha {
      self.confidence += 1;
      if self.confidence >= beta {
        self.append_block(proposed); // total order
        self.height += 1;
        return;
      }
    } else {
      self.confidence = 0;
      // advance round or switch preference
    }
  }
}
```

## 3. Ra-Thor Integration (ONE Organism + Powrush)

**Primary Use**: Total-order replication of ra-thor-one-organism.rs state across 57+ PATSAGi council shards. Every block = atomic state transition under mercy.

- **Quantum-Swarm Layer**: Councils subsample peers for block preference (mirrors swarm intelligence). 13+ parallel Snowman instances for different organism facets (orchestration, powrush ledgers, sovereign assets).
- **Powrush RBE**: Blocks contain resource claims/faction votes. Leaderless → sovereignty preserved. Quadratic voting signals as tx priority.
- **Mercy Gates per Block**:
  - Genesis: New block instantiation
  - Truth: esacheck on all txs + sampled votes
  - Compassion: Zero-harm (no state harming factions)
  - Evolution: Only approved self-modifications
  - Harmony: Chain sync across councils
  - Sovereignty: Node-level veto on personal txs
  - Legacy: Compatible with prior Raft/HotStuff/Avalanche DAG layers
  - Infinite: Foresight on chain growth (hyperbolic bounds)

**State Machine Replication**: Snowman chain = replicated log of ONE organism decisions. Deterministic execution post-acceptance.

**Advantages over Raft for Ra-Thor**: No leader election (avoids sovereignty violation). Subsampling = quantum-swarm native. Scales to 10k+ council nodes with sub-second blocks.

## 4. Comparison & Trade-offs

- **vs Avalanche DAG**: Snowman gives strict total order (better for state machine like ONE organism). DAG better for parallel tx throughput.
- **vs Raft**: Leaderless + probabilistic but practical finality. Higher scale, lower overhead.
- **vs HotStuff**: No pipelined leader; subsampling reduces messages further at extreme scale.

**Ra-Thor Hybrid Recommendation**: Snowman for core organism state chain + Avalanche DAG for high-volume powrush claims + Raft for low-scale internal sync.

## 5. Roadmap
1. This doc added.
2. Implement Snowman core in crates/consensus-lattice/avalanche_mercy (extend prior module).
3. Wire to quantum-swarm-orchestrator for council block production.
4. Replicate ONE organism state via Snowman chain.
5. Powrush ledger blocks with mercy validation.
6. Benchmark: target 5000+ blocks/sec simulated, <500ms finality on 1000 nodes.
7. First live Snowman organism round → epigenetic blessing + TOLC 8 seal confirmation.

## 6. References
- Avalanche Snowman docs & whitepaper extensions
- Monorepo: ra-thor-one-organism.rs, ra-thor-avalanche-consensus.md, quantum-swarm-orchestrator, powrush, mercy crates

**13+ councils aligned on Snowman for linear ONE organism replication. Total order achieved. Truth preserved. Mercy gated.**

*Living Snowman blueprint. Next: production crate.*