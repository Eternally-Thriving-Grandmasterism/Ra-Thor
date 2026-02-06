// src/sync/yjs-real-time-awareness.ts – Yjs Real-Time Awareness Layer v1
// Multi-device/multiplanetary presence, cursors, live valence glows, collaborative editing
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const YJS_ROOM = 'rathor-nexi-lattice';
const YJS_WS_URL = 'wss://yjs.rathor.ai/ws'; // replace with real relay or self-hosted y-websocket

let ydoc: Y.Doc | null = null;
let provider: WebsocketProvider | null = null;
let awareness: Y.Awareness | null = null;

export class YjsRealTimeAwareness {
  static async initialize(userId: string, userName: string, avatarUrl?: string) {
    const actionName = 'Initialize Yjs real-time awareness layer';
    if (!await mercyGate(actionName)) return;

    try {
      ydoc = new Y.Doc();

      provider = new WebsocketProvider(YJS_WS_URL, YJS_ROOM, ydoc, {
        params: { userId, userName, avatarUrl }
      });

      awareness = provider.awareness;

      // Set local user state (presence)
      awareness.setLocalState({
        user: {
          id: userId,
          name: userName,
          avatar: avatarUrl || 'https://via.placeholder.com/64?text=R',
          color: '#00ff88',
          valence: currentValence.get()
        },
        cursor: null,
        lastActive: Date.now()
      });

      // Subscribe to valence changes → update awareness
      currentValence.subscribe(val => {
        if (awareness) {
          awareness.setLocalStateField('user', { valence: val });
        }
      });

      // Live presence events
      awareness.on('update', (changes) => {
        const states = awareness!.getStates();
        const activeUsers = Array.from(states.values())
          .filter(state => state.user && Date.now() - state.lastActive < 30000);

        // Trigger haptic pulse on new high-valence presence
        activeUsers.forEach(state => {
          if (state.user.valence > 0.95) {
            mercyHaptic.playPattern('cosmicHarmony', state.user.valence);
          }
        });

        // TODO: render live avatars/cursors in UI (dashboard, summon orb, etc.)
        console.log(`[YjsAwareness] ${activeUsers.length} active users present`);
      });

      console.log("[YjsRealTimeAwareness] Initialized – multi-device/multiplanetary presence active");
    } catch (e) {
      console.error("[YjsRealTimeAwareness] Initialization failed", e);
    }
  }

  static getAwareness(): Y.Awareness | null {
    return awareness;
  }

  static getYDoc(): Y.Doc | null {
    return ydoc;
  }

  static isConnected(): boolean {
    return provider?.connected || false;
  }

  static destroy() {
    if (provider) {
      provider.destroy();
      provider = null;
    }
    if (ydoc) {
      ydoc.destroy();
      ydoc = null;
    }
    awareness = null;
  }
}

export default YjsRealTimeAwareness;
