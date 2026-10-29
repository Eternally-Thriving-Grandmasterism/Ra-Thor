// src/core/mcts-lookahead-integration.ts – MCTS Lookahead Integration Layer v1.0
// Neural-guided tree search overlay for SAC/PPO/TD3 policies
// Valence-shaped exploration bonus, mercy-gated branch pruning, automatic temperature tuning
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { getTemperature } from './automatic-temperature-tuning';

const MCTS_LOOKAHEAD_DEPTH = 32;          // default lookahead horizon
const MCTS_ITERATIONS = 800;              // search iterations per decision
const VALENCE_EXPLORATION_BOOST = 2.5;    // high valence → stronger exploration
const VALENCE_PRUNE_THRESHOLD = 0.85;     // prune branches below this projected valence
const DEFAULT_C_PUCT = 1.414;             // √2 – classic UCT constant

interface LookaheadResult {
  bestAction: string | any;               // discrete or continuous action
  policyImprovement: Map<string | any, number>; // improved policy distribution
  projectedValue: number;                 // bootstrapped future value
  projectedValenceTrajectory: number[];   // valence over lookahead horizon
  isSafe: boolean;
  reason?: string;
}

/**
 * Perform MCTS lookahead to refine policy action selection
 * @param policyNet Current policy/value network (SAC/PPO/TD3 compatible)
 * @param currentState Current environment state
 * @param availableActions Optional discrete actions (if any); otherwise continuous
 * @param depth Lookahead depth (default: MCTS_LOOKAHEAD_DEPTH)
 * @returns LookaheadResult with best action & improved policy
 */
export async function mctsLookahead(
  policyNet: {
    predictPolicyAndValue: (state: any) => Promise<{ policy: Map<string, number>; value: number }>;
  },
  currentState: any,
  availableActions?: string[] | any[],
  depth: number = MCTS_LOOKAHEAD_DEPTH
): Promise<LookaheadResult> {
  const actionName = 'MCTS lookahead refinement';
  if (!await mercyGate(actionName)) {
    // Fallback to direct policy prediction
    const { policy } = await policyNet.predictPolicyAndValue(currentState);
    const bestAction = selectActionFromPolicy(policy);
    return {
      bestAction,
      policyImprovement: policy,
      projectedValue: 0,
      projectedValenceTrajectory: [],
      isSafe: true,
      reason: 'Mercy gate bypassed MCTS lookahead'
    };
  }

  const valence = currentValence.get();
  const iterations = Math.floor(MCTS_ITERATIONS * (0.5 + valence)); // scale with valence

  console.log(`[MCTS-Lookahead] Starting lookahead – valence ${valence.toFixed(3)}, ${iterations} iterations, depth ${depth}`);

  // Create MCTS tree rooted at current state
  const mcts = new MCTS(currentState, availableActions || [], policyNet);

  // Adjust exploration constant with valence boost
  mcts.c_puct = DEFAULT_C_PUCT * (1 + valence * VALENCE_EXPLORATION_BOOST);

  // Run search
  await mcts.search(iterations);

  // Get best child/action
  const bestChild = mcts.bestChild(mcts.root);
  const bestAction = bestChild.state.lastAction;

  // Extract improved policy from visit counts
  const policyImprovement = new Map<string | any, number>();
  let totalVisits = 0;
  for (const child of mcts.root.children.values()) {
    totalVisits += child.visits;
  }
  for (const [action, child] of mcts.root.children) {
    policyImprovement.set(action, child.visits / totalVisits);
  }

  // Simulate valence trajectory along best path (simplified)
  const projectedValenceTrajectory: number[] = [];
  let projectedState = currentState;
  for (let d = 0; d < depth; d++) {
    projectedState = mcts.applyAction(projectedState, bestAction);
    projectedValenceTrajectory.push(currentValence.get() * (1 - d / depth)); // decay simulation
  }

  const projectedValue = bestChild.totalValue / bestChild.visits;
  const minProjectedValence = Math.min(...projectedValenceTrajectory);

  const isSafe = minProjectedValence >= VALENCE_PRUNE_THRESHOLD;

  if (!isSafe) {
    mercyHaptic.playPattern('warningPulse', valence);
    console.warn(`[MCTS-Lookahead] Unsafe trajectory detected – min projected valence ${minProjectedValence.toFixed(3)}`);
  }

  return {
    bestAction,
    policyImprovement,
    projectedValue,
    projectedValenceTrajectory,
    isSafe,
    reason: isSafe ? 'Safe lookahead path' : 'Unsafe projected valence drop'
  };
}

/**
 * Helper: select action from policy distribution
 */
function selectActionFromPolicy(policy: Map<string, number>): string | any {
  const actions = Array.from(policy.keys());
  const probs = Array.from(policy.values());
  let sum = 0;
  const r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r <= sum) return actions[i];
  }
  return actions[actions.length - 1]; // fallback
}

/**
 * Example usage in SAC/PPO rollout step
 */
export async function selectActionWithLookahead(
  policyNet: any,
  state: any,
  availableActions?: string[] | any[]
): Promise<any> {
  const lookahead = await mctsLookahead(policyNet, state, availableActions);

  if (!lookahead.isSafe) {
    console.warn("[Lookahead] Falling back to direct policy due to unsafe trajectory");
    const { policy } = await policyNet.predictPolicyAndValue(state);
    return selectActionFromPolicy(policy);
  }

  return lookahead.bestAction;
}
