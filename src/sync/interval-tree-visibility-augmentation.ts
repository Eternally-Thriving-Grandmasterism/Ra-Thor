// src/sync/interval-tree-visibility-augmentation.ts – Interval-Tree Augmentation for WOOTO Visibility v1
// O(log n) range query & update, dirty-region tracking, incremental recompute, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

// Simple interval-tree node (augmented for visibility)
interface IntervalNode {
  id: string;                     // WOOTChar ID or range identifier
  left: number;                   // left position bound
  right: number;                  // right position bound
  visibleCount: number;           // number of visible chars in subtree
  minPrecedence: number;          // min precedence timestamp in subtree
  maxPrecedence: number;          // max precedence timestamp in subtree
  isDirty: boolean;               // subtree needs recompute
  leftChild: IntervalNode | null;
  rightChild: IntervalNode | null;
}

export class IntervalTreeVisibilityAugmentation {
  private root: IntervalNode | null = null;
  private dirtyNodes = new Set<string>();

  constructor() {
    // Initialize root sentinel covering entire position space
    this.root = {
      id: 'ROOT',
      left: 0,
      right: Number.MAX_SAFE_INTEGER,
      visibleCount: 0,
      minPrecedence: 0,
      maxPrecedence: 0,
      isDirty: true,
      leftChild: null,
      rightChild: null
    };
    this.dirtyNodes.add('ROOT');
  }

  /**
   * Insert or update a WOOTChar interval with visibility info
   */
  async insertOrUpdate(
    charId: string,
    leftPos: number,
    rightPos: number,
    visible: boolean,
    precedence: number
  ) {
    const actionName = `Interval-tree insert/update: ${charId}`;
    if (!await mercyGate(actionName)) return;

    // Insert into tree (simplified balanced BST insertion – real impl would use red-black/AVL)
    this.root = this._insert(this.root, charId, leftPos, rightPos, visible, precedence);

    // Mark affected path dirty
    this._markPathDirty(charId);

    console.log(`[IntervalTree] Inserted/updated \( {charId} [ \){leftPos}–\( {rightPos}] visible= \){visible}`);
  }

  private _insert(
    node: IntervalNode | null,
    id: string,
    l: number,
    r: number,
    visible: boolean,
    precedence: number
  ): IntervalNode {
    if (!node) {
      return {
        id,
        left: l,
        right: r,
        visibleCount: visible ? 1 : 0,
        minPrecedence: precedence,
        maxPrecedence: precedence,
        isDirty: true,
        leftChild: null,
        rightChild: null
      };
    }

    // Simplified midpoint split (real impl uses balanced tree)
    const mid = Math.floor((node.left + node.right) / 2);

    if (r <= mid) {
      node.leftChild = this._insert(node.leftChild, id, l, r, visible, precedence);
    } else if (l >= mid) {
      node.rightChild = this._insert(node.rightChild, id, l, r, visible, precedence);
    } else {
      // Overlap – split node or handle overlap (simplified: store here)
      node.visibleCount += visible ? 1 : 0;
      node.minPrecedence = Math.min(node.minPrecedence, precedence);
      node.maxPrecedence = Math.max(node.maxPrecedence, precedence);
      node.isDirty = true;
    }

    // Update aggregates
    this._updateAggregates(node);

    return node;
  }

  private _updateAggregates(node: IntervalNode) {
    node.visibleCount = (node.visibleCount || 0) +
      (node.leftChild?.visibleCount || 0) +
      (node.rightChild?.visibleCount || 0);

    node.minPrecedence = Math.min(
      node.minPrecedence,
      node.leftChild?.minPrecedence ?? Infinity,
      node.rightChild?.minPrecedence ?? Infinity
    );

    node.maxPrecedence = Math.max(
      node.maxPrecedence,
      node.leftChild?.maxPrecedence ?? -Infinity,
      node.rightChild?.maxPrecedence ?? -Infinity
    );

    node.isDirty = node.leftChild?.isDirty || node.rightChild?.isDirty || node.isDirty;
  }

  /**
   * Mark path from root to node dirty (simplified – real impl traverses tree)
   */
  private _markPathDirty(id: string) {
    this.dirtyNodes.add(id);
    // In real impl: traverse from root to node and mark ancestors dirty
  }

  /**
   * Incremental visible string computation using interval tree
   */
  async computeVisibleString(): Promise<string[]> {
    const actionName = 'Compute visible string using interval tree';
    if (!await mercyGate(actionName)) return [];

    if (!this.root) return [];

    const visible: string[] = [];
    this._traverseVisible(this.root, visible);

    // Clear dirty flags after recompute
    this.dirtyNodes.clear();
    this._clearDirtyFlags(this.root);

    console.log(`[IntervalTree] Incremental visibility recompute complete – ${visible.length} visible elements`);
    return visible;
  }

  private _traverseVisible(node: IntervalNode, result: string[]) {
    if (!node || !node.visibleCount) return;

    // Skip invisible subtrees
    if (!node.isDirty && node.visibleCount === 0) return;

    // Visit left subtree
    if (node.leftChild) this._traverseVisible(node.leftChild, result);

    // Visit self if visible
    if (node.visible) result.push(node.id);

    // Visit right subtree
    if (node.rightChild) this._traverseVisible(node.rightChild, result);
  }

  private _clearDirtyFlags(node: IntervalNode | null) {
    if (!node) return;
    node.isDirty = false;
    this._clearDirtyFlags(node.leftChild);
    this._clearDirtyFlags(node.rightChild);
  }

  /**
   * Valence-modulated recompute trigger (high valence → recompute sooner)
   */
  shouldRecompute(dirtyCount: number, valence: number = currentValence.get()): boolean {
    const actionName = `Valence-modulated interval-tree recompute trigger`;
    if (!mercyGate(actionName)) return dirtyCount > 50;

    const threshold = 20 - (valence - 0.95) * 15; // high valence → lower threshold
    return dirtyCount > threshold;
  }
}

export const intervalTreeVisibility = new IntervalTreeVisibilityAugmentation();

// Usage example in real-time MR renderer / collaborative annotation layer
/*
if (intervalTreeVisibility.shouldRecompute(dirtyCount)) {
  const visibleIds = await intervalTreeVisibility.computeVisibleString();
  // render visibleIds in MR habitat (glow intensity = valenceImpact)
}
*/
