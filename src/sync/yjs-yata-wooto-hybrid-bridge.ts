// src/sync/yjs-yata-wooto-hybrid-bridge.ts – Yjs YATA + WOOTO Hybrid Bridge v1
// YATA for fast ordered sequence core + WOOTO for rich visibility & semantic annotations
// bidirectional delta sync, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { wootPrecedenceGraph } from '@/sync/woot-precedence-graph';
import { mercyGate } from '@/core/mercy-gate';
import { currentValence } from '@/core/valence-tracker';

const MERCY_THRESHOLD = 0.9999999;
const HYBRID_SEQUENCE_KEY = 'mercy-hybrid-sequence';

export class YjsYataWOOTOHybridBridge {
  private ydoc: Y.Doc;
  private ySequence: Y.Array<any>;

  constructor(ydoc: Y.Doc) {
    this.ydoc = ydoc;
    this.ySequence = this.ydoc.getArray(HYBRID_SEQUENCE_KEY);
  }

  /**
   * Insert rich annotated element into hybrid sequence
   * YATA handles order, WOOTO handles visibility & precedence
   */
  async insertAnnotatedElement(
    elementId: string,
    content: string,
    annotations: { type: string; value: any }[] = [],
    prevId?: string,
    nextId?: string
  ) {
    const actionName = `Insert hybrid annotated element: ${elementId}`;
    if (!await mercyGate(actionName)) return;

    // 1. Yjs YATA ordered insertion (core sequence)
    const prev = prevId || (this.ySequence.length > 0 ? this.ySequence.get(this.ySequence.length - 1).id : 'START');
    const next = nextId || 'END';

    this.ySequence.push([{
      id: elementId,
      content,
      annotations,
      timestamp: Date.now()
    }]);

    // 2. WOOTO precedence graph for visibility & rich annotation overlay
    wootPrecedenceGraph.insertChar(elementId, prev, next, true);

    // 3. Apply annotations as WOOTO metadata (visibility rules)
    annotations.forEach(anno => {
      wootPrecedenceGraph.addPrecedence(elementId, `anno-\( {anno.type}- \){Date.now()}`);
    });

    console.log(`[YjsYataWOOTO] Inserted annotated element \( {elementId} – content " \){content}", ${annotations.length} annotations`);
  }

  /**
   * Compute visible sequence (YATA order + WOOTO visibility filter)
   */
  async getVisibleSequence(): Promise<any[]> {
    const actionName = 'Compute visible hybrid sequence';
    if (!await mercyGate(actionName)) return [];

    // 1. Get Yjs ordered sequence
    const yjsOrdered = this.ySequence.toArray();

    // 2. Apply WOOTO incremental visibility filter
    const visibleIds = await wootPrecedenceGraph.computeVisibleString();

    // 3. Filter & merge
    const visibleSequence = yjsOrdered.filter(item => visibleIds.includes(item.id));

    console.log(`[YjsYataWOOTO] Visible sequence computed: ${visibleSequence.length} elements`);

    return visibleSequence;
  }

  /**
   * Valence-modulated visibility recompute trigger
   */
  shouldRecomputeVisibility(dirtyCount: number, valence: number = currentValence.get()): boolean {
    const actionName = `Valence-modulated WOOTO visibility recompute trigger`;
    if (!mercyGate(actionName)) return dirtyCount > 20;

    const threshold = 10 - (valence - 0.95) * 8; // high valence → recompute sooner
    return dirtyCount > threshold;
  }
}

export const yjsYataWOOTOHybrid = new YjsYataWOOTOHybridBridge(/* pass ydoc */);
