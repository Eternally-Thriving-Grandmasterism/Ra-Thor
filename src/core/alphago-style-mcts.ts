// src/core/alphago-style-mcts.ts – AlphaGo-style MCTS Engine v1.0
// Neural-guided UCT + value/policy heads + valence-weighted exploration + mercy gating
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

// ──────────────────────────────────────────────────────────────
// Neural Network Interface (placeholder – replace with real model)
// ──────────────────────────────────────────────────────────────

interface NeuralNetwork {
  predict(state: any): Promise<{
    policy: Map<string, number>;   // action → probability
    value: number;                 // estimated win probability / valence reward [0,1]
  }>;
}

// ──────────────────────────────────────────────────────────────
// Node structure
// ──────────────────────────────────────────────────────────────

interface MCTSNode {
  state: any;
  parent: MCTSNode | null;
  children: Map<string, MCTSNode>;
  visits: number;
  totalValue: number;             // sum of value estimates
  untriedActions: string[];
  isTerminal: boolean;
  depth: number;
}

export class AlphaGoStyleMCTS {
  private root: MCTSNode;
  private neuralNet: NeuralNetwork;
  private c_puct: number = 1.0;   // exploration constant (AlphaGo Zero used \~1.0–5.0)
  private dirichletNoiseAlpha: number = 0.3;
  private maxIterations: number = 1600;
  private maxDepth: number = 128;

  constructor(initialState: any, initialActions: string[], neuralNet: NeuralNetwork) {
    this.neuralNet = neuralNet;
    this.root = this.createNode(initialState, null, initialActions);
  }

  private createNode(state: any, parent: MCTSNode | null, actions: string[]): MCTSNode {
    return {
      state,
      parent,
      children: new Map(),
      visits: 0,
      totalValue: 0,
      untriedActions: [...actions],
      isTerminal: false,
      depth: parent ? parent.depth + 1 : 0
    };
  }

  /**
   * Run full AlphaGo-style MCTS search and return best action + policy
   */
  async search(): Promise<{ bestAction: string; policy: Map<string, number> }> {
    const actionName = 'Run AlphaGo-style MCTS search';
    if (!await mercyGate(actionName)) {
      return { bestAction: this.root.untriedActions[0] || 'none', policy: new Map() };
    }

    const valence = currentValence.get();
    const iterations = Math.floor(this.maxIterations * (0.5 + valence * 0.5)); // scale with valence

    console.log(`[AlphaGoMCTS] Starting search – valence ${valence.toFixed(3)}, ${iterations} iterations`);

    // Add Dirichlet noise at root for exploration
    this.addDirichletNoise(this.root);

    for (let i = 0; i < iterations; i++) {
      const path = this.select();
      const leaf = path[path.length - 1];
      const { policy, value } = await this.evaluate(leaf);
      this.expand(leaf, policy);
      this.backpropagate(path, value);
    }

    const bestChild = this.bestChild(this.root);
    const bestAction = bestChild.state.lastAction || 'none';

    // Extract improved policy (visit counts)
    const policy = new Map<string, number>();
    for (const [action, child] of this.root.children) {
      policy.set(action, child.visits / this.root.visits);
    }

    mercyHaptic.playPattern(valence > 0.9 ? 'cosmicHarmony' : 'neutralPulse', valence);
    console.log(`[AlphaGoMCTS] Best action: ${bestAction} (visits: ${bestChild.visits}, Q: ${(bestChild.totalValue / bestChild.visits).toFixed(3)})`);

    return { bestAction, policy };
  }

  /**
   * Selection phase – PUCT (Predictor + UCT)
   */
  private select(): MCTSNode[] {
    const path: MCTSNode[] = [];
    let node = this.root;

    while (node.untriedActions.length === 0 && node.children.size > 0 && !node.isTerminal) {
      path.push(node);
      node = this.bestChild(node);
    }

    path.push(node);
    return path;
  }

  /**
   * PUCT formula with valence bonus
   */
  private bestChild(node: MCTSNode): MCTSNode {
    const valence = currentValence.get();
    const c_puct_valence = this.c_puct * (1 + valence * 2.0); // stronger exploration on high valence

    let bestChild: MCTSNode | null = null;
    let bestPUCT = -Infinity;

    for (const child of node.children.values()) {
      const q = child.visits > 0 ? child.totalValue / child.visits : 0;
      const u = c_puct_valence * Math.sqrt(node.visits) / (1 + child.visits);
      const puct = q + u;

      if (puct > bestPUCT) {
        bestPUCT = puct;
        bestChild = child;
      }
    }

    return bestChild!;
  }

  /**
   * Expansion phase – create new nodes from untried actions
   */
  private async expand(node: MCTSNode, policy: Map<string, number>) {
    if (node.isTerminal || node.untriedActions.length === 0) return;

    const action = node.untriedActions.shift()!;
    const nextState = this.applyAction(node.state, action);
    const child = this.createNode(nextState, node, this.getActions(nextState));
    child.state.lastAction = action;
    node.children.set(action, child);
  }

  /**
   * Simulation / evaluation – use neural network value head
   */
  private async evaluate(node: MCTSNode): Promise<{ policy: Map<string, number>; value: number }> {
    if (node.isTerminal) {
      return { policy: new Map(), value: this.evaluateTerminal(node.state) };
    }

    const { policy, value } = await this.neuralNet.predict(node.state);

    // Add Dirichlet noise for root exploration (only at root)
    if (node === this.root) {
      this.addDirichletNoiseToPolicy(policy);
    }

    return { policy, value };
  }

  private addDirichletNoiseToPolicy(policy: Map<string, number>) {
    const noise = new Map<string, number>();
    const alpha = this.dirichletNoiseAlpha;
    let sum = 0;

    for (const action of policy.keys()) {
      const n = gamma(alpha); // Dirichlet sample (simplified)
      noise.set(action, n);
      sum += n;
    }

    for (const [action, p] of policy) {
      const n = noise.get(action)! / sum;
      policy.set(action, 0.75 * p + 0.25 * n);
    }
  }

  private evaluateTerminal(state: any): number {
    // Terminal reward = final valence projection or win/loss
    return state.finalValence || currentValence.get();
  }

  /**
   * Backpropagation – update value & visits
   */
  private backpropagate(path: MCTSNode[], value: number) {
    for (const node of path.reverse()) {
      node.visits++;
      node.totalValue += value;
    }
  }

  // ─── Abstract methods – must be implemented by concrete planner ───
  protected getActions(state: any): string[] {
    throw new Error("getActions not implemented");
  }

  protected applyAction(state: any, action: string): any {
    throw new Error("applyAction not implemented");
  }

  // ─── Concrete planner example: negotiation tree ───
  static createNegotiationPlanner(currentState: any) {
    return new (class extends MCTS {
      protected getActions(state: any): string[] {
        return ['propose-alliance', 'offer-resources', 'request-aid', 'reject', 'wait'];
      }

      protected applyAction(state: any, action: string): any {
        return { ...state, lastAction: action, valence: state.valence * 1.02 }; // simplified
      }
    })(currentState, ['propose-alliance', 'offer-resources', 'request-aid', 'reject', 'wait'], new MockNeuralNet());
  }
}

// Mock neural net for demo (replace with real WebLLM / tfjs model)
class MockNeuralNet implements NeuralNetwork {
  async predict(state: any) {
    return {
      policy: new Map([
        ['propose-alliance', 0.4],
        ['offer-resources', 0.25],
        ['request-aid', 0.15],
        ['reject', 0.1],
        ['wait', 0.1]
      ]),
      value: currentValence.get()
    };
  }
}

export default MCTS;
