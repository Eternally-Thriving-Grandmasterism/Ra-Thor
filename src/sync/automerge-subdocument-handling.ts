// src/sync/automerge-subdocument-handling.ts – Automerge Subdocument Handling Manager v1
// Binary embedding for true independent subdocuments, lazy load, GC safety, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

export class AutomergeSubdocumentHandler {
  private parentDoc: Automerge.Doc<any>;
  private subdocCache = new Map<string, Automerge.Doc<any>>();

  constructor(parentDoc: Automerge.Doc<any>) {
    this.parentDoc = parentDoc;
  }

  /**
   * Get or create subdocument via binary embedding (lazy + mercy-gated)
   */
  async getOrCreate(
    key: string,
    initialValue: any = {},
    requiredValence: number = MERCY_THRESHOLD
  ): Promise<Automerge.Doc<any> | null> {
    const actionName = `Automerge subdoc get/create: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) {
      return null;
    }

    // Check cache first
    if (this.subdocCache.has(key)) {
      return this.subdocCache.get(key)!;
    }

    // Check if already embedded in parent
    const parentMap = Automerge.get(this.parentDoc, ['subdocs']) || {};
    const binary = parentMap[key];

    let subdoc: Automerge.Doc<any>;

    if (binary && binary instanceof Uint8Array) {
      try {
        subdoc = Automerge.load(binary);
        console.log(`[AutomergeSubdoc] Loaded embedded subdoc from binary: \( {key} ( \){binary.byteLength} bytes)`);
      } catch (e) {
        console.warn(`[AutomergeSubdoc] Failed to load binary for ${key}`, e);
        subdoc = Automerge.from(initialValue);
      }
    } else {
      subdoc = Automerge.from(initialValue);
      console.log(`[AutomergeSubdoc] Created new subdoc: ${key}`);
    }

    this.subdocCache.set(key, subdoc);
    return subdoc;
  }

  /**
   * Save subdoc back into parent as binary blob (compact serialization)
   */
  async save(key: string, requiredValence: number = MERCY_THRESHOLD) {
    const actionName = `Automerge subdoc save: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) return;

    const subdoc = this.subdocCache.get(key);
    if (!subdoc) return;

    const binary = Automerge.save(subdoc);

    // Update parent document
    Automerge.change(this.parentDoc, `Saving subdoc ${key}`, doc => {
      if (!doc.subdocs) doc.subdocs = {};
      doc.subdocs[key] = binary;
    });

    console.log(`[AutomergeSubdoc] Subdoc saved as binary: \( {key} ( \){binary.byteLength} bytes)`);
  }

  /**
   * Destroy subdoc (remove from parent & clear cache)
   */
  async destroy(key: string) {
    Automerge.change(this.parentDoc, `Destroying subdoc ${key}`, doc => {
      if (doc.subdocs) delete doc.subdocs[key];
    });

    this.subdocCache.delete(key);
    console.log(`[AutomergeSubdoc] Subdoc destroyed: ${key}`);
  }

  /**
   * Get all cached subdocs (for monitoring / bulk sync)
   */
  getCachedSubdocs(): Map<string, Automerge.Doc<any>> {
    return new Map(this.subdocCache);
  }
}

export const automergeSubdocHandler = new AutomergeSubdocumentHandler(/* pass global Automerge doc */);
