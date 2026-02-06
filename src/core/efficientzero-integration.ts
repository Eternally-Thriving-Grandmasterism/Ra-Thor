// src/core/efficientzero-integration.ts – EfficientZero Model Integration Layer v1.0
// EfficientZero-style model-based RL: representation, dynamics, prediction + self-supervised consistency
// Valence-weighted planning priority, mercy-gated simulation depth, lattice-integrated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import MCTS from './alphago-style-mcts-neural';

const EFFICIENTZERO_PLANNING_DEPTH = 50;          // max simulated steps in planning
const EFFICIENTZERO_MCTS_ITERATIONS = 800;        // MCTS search budget per decision
const VALENCE_PLANNING_BOOST = 2.5;               // high valence → deeper planning & more iterations
const VALENCE_PRUNE_THRESHOLD = 0.85;             // prune simulated paths below this projected valence
const CONSISTENCY_LOSS_COEF = 0.5;                // self-supervised consistency loss weight
const REWARD_LOSS_COEF = 0.3;                     // reward prediction auxiliary loss
const VALUE_LOSS_COEF = 0.2;                      // value prediction loss
const MIN_PLANNING_DEPTH = 5;
const MAX_PLANNING_DEPTH = 100;

interface EfficientZeroNetworks {
  representation: (observation: any) => Promise<any>;           // o → hidden state h
  dynamics: (hidden: any, action: any) => Promise<{
    nextHidden: any;
    reward: number;
    done: boolean;
  }>; // (h, a) → (h', r, done)
  prediction: (hidden: any) => Promise<{
    policy: Map<string, number>;
    value: number;
  }>; // h → (p, v)
  consistency: (hidden: any) => Promise<any>;                   // projection head for self-supervised loss
}

interface SimulatedNode {
  hiddenState: any;
  reward: number;
  done: boolean;
  policy: Map<string, number>;
  value: number;
  children: Map<string, SimulatedNode>;
  visits: number;
  totalValue: number;
  depth: number;
  isTerminal: boolean;
}

export class EfficientZeroIntegration {
  private networks: EfficientZeroNetworks;

  constructor(networks: EfficientZeroNetworks) {
    this.networks = networks;
  }

  /**
   * Perform EfficientZero-style planning: build search tree with learned model
   * @param initialObservation Root observation (not hidden state)
   * @returns Best action & improved policy
   */
  async plan(initialObservation: any): Promise<{ bestAction: string | any; policy: Map<string, number> }> {
    const actionName = 'EfficientZero model-based planning';
    if (!await mercyGate(actionName)) {
      // Fallback to direct policy prediction
      const hidden = await this.networks.representation(initialObservation);
      const { policy } = await this.networks.prediction(hidden);
      const bestAction = selectActionFromPolicy(policy);
      return { bestAction, policy };
    }

    const valence = currentValence.get();
    const planningDepth = Math.floor(MIN_PLANNING_DEPTH + (MAX_PLANNING_DEPTH - MIN_PLANNING_DEPTH) * valence);
    const iterations = Math.floor(EFFICIENTZERO_MCTS_ITERATIONS * (0.5 + valence * VALENCE_PLANNING_BOOST));

    console.log(`[EfficientZero] Planning start – valence ${valence.toFixed(3)}, depth ${planningDepth}, ${iterations} iterations`);

    // 1. Initial hidden state from representation network
    const rootHidden = await this.networks.representation(initialObservation);

    // 2. Root prediction
    const { policy: rootPolicy, value: rootValue } = await this.networks.prediction(rootHidden);

    const root: SimulatedNode = {
      hiddenState: rootHidden,
      reward: 0,
      done: false,
      policy: rootPolicy,
      value: rootValue,
      children: new Map(),
      visits: 0,
      totalValue: 0,
      depth: 0,
      isTerminal: false
    };

    // 3. Run MCTS with learned model
    for (let i = 0; i < iterations; i++) {
      const path = this.select(root);
      const leaf = path[path.length - 1];

      // Dynamics step
      const action = leaf.state.lastAction || selectActionFromPolicy(leaf.policy);
      const { nextHidden, reward, done } = await this.networks.dynamics(leaf.hiddenState, action);

      // Prediction step
      const { policy, value } = await this.networks.prediction(nextHidden);

      // Consistency regularization (self-supervised – project hidden states)
      const projected = await this.networks.consistency(leaf.hiddenState);
      const consistencyLoss = computeConsistencyLoss(projected, nextHidden); // cosine or MSE

      // Mercy gate: prune low-valence paths
      if (value < VALENCE_PRUNE_THRESHOLD && leaf.depth > 3) {
        continue;
      }

      this.expand(leaf, policy, reward, nextHidden, done);
      this.backpropagate(path, value);
    }

    // 4. Extract best action & improved policy
    const bestChild = this.bestChild(root);
    const bestAction = bestChild.state.lastAction;

    const policyImprovement = new Map<string, number>();
    let totalVisits = 0;
    for (const child of root.children.values()) {
      totalVisits += child.visits;
    }
    for (const [action, child] of root.children) {
      policyImprovement.set(action, child.visits / totalVisits);
    }

    const projectedValue = bestChild.totalValue / bestChild.visits;

    mercyHaptic.playPattern(valence > 0.9 ? 'cosmicHarmony' : 'neutralPulse', valence);

    return {
      bestAction,
      policy: policyImprovement,
      projectedValue
    };
  }

  private select(root: SimulatedNode): SimulatedNode[] {
    const path: SimulatedNode[] = [];
    let node = root;

    while (node.children.size > 0 && !node.isTerminal) {
      path.push(node);
      node = this.bestChild(node);
    }

    path.push(node);
    return path;
  }

  private bestChild(node: SimulatedNode): SimulatedNode {
    const valence = currentValence.get();
    const c_puct = DEFAULT_C_PUCT * (1 + valence * VALENCE_EXPLORATION_BOOST);

    let bestChild: SimulatedNode | null = null;
    let bestPUCT = -Infinity;

    for (const child of node.children.values()) {
      const q = child.visits > 0 ? child.totalValue / child.visits : 0;
      const u = c_puct * child.policy.get(child.state.lastAction || '')! * Math.sqrt(node.visits) / (1 + child.visits);
      const puct = q + u;

      if (puct > bestPUCT) {
        bestPUCT = puct;
        bestChild = child;
      }
    }

    return bestChild!;
  }

  private expand(parent: SimulatedNode, policy: Map<string, number>, reward: number, nextHidden: any, done: boolean) {
    for (const [action, prior] of policy) {
      if (!parent.children.has(action)) {
        const child: SimulatedNode = {
          hiddenState: nextHidden,
          reward,
          done,
          policy,
          value: 0,
          children: new Map(),
          visits: 0,
          totalValue: 0,
          depth: parent.depth + 1,
          isTerminal: done
        };
        parent.children.set(action, child);
      }
    }
  }

  private backpropagate(path: SimulatedNode[], value: number) {
    for (const node of path.reverse()) {
      node.visits++;
      node.totalValue += value;
    }
  }
}

function computeConsistencyLoss(proj: any, target: any): number {
  // Cosine similarity or MSE between projection & target hidden states
  // Placeholder – real impl depends on hidden representation
  return 0;
}

function selectActionFromPolicy(policy: Map<string, number>): string {
  const actions = Array.from(policy.keys());
  const probs = Array.from(policy.values());
  let sum = 0;
  const r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r <= sum) return actions[i];
  }
  return actions[actions.length - 1];
}

export default EfficientZeroIntegration;
