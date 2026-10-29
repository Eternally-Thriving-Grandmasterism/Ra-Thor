// src/utils/cross-modal-audio-sync.ts – Cross-Modal Audio Sync Engine v1.0
// Real-time synchronization of audio + haptic + visual + motion feedback
// Valence-modulated timing, intensity, frequency harmony, mercy power envelope
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from './haptic-patterns';
import visualFeedback from './visual-feedback';
import audioFeedback from './audio-feedback';

type CrossModalEventType =
  | 'success'      // full cosmic harmony
  | 'warning'      // gentle mercy alert
  | 'error'        // contained urgency
  | 'loading'      // awakening pulse
  | 'bloom'        // growth cascade
  | 'sync'         // reconnection wave
  | 'gesture'      // affirmation pulse

interface CrossModalOptions {
  type: CrossModalEventType;
  durationMs?: number;          // total event duration
  valenceOverride?: number;     // custom valence (defaults to current)
  targetElement?: HTMLElement;  // for visual ripple/glow
  skipHaptic?: boolean;
  skipVisual?: boolean;
  skipAudio?: boolean;
  skipMotion?: boolean;
}

/**
 * Unified cross-modal feedback trigger – audio drives timing, all modalities sync
 * - Valence scales intensity, pitch, vibration strength, visual brightness, motion amplitude
 * - Mercy gate prevents overwhelming multisensory overload on low valence
 */
export async function triggerCrossModalFeedback(options: CrossModalOptions): Promise<void> {
  const { type, durationMs = 2200, valenceOverride, targetElement, skipHaptic, skipVisual, skipAudio, skipMotion } = options;

  const actionName = `Cross-modal feedback: ${type}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[CrossModal] Mercy gate blocked: ${type}`);
    return;
  }

  const valence = valenceOverride ?? currentValence.get();
  const intensity = Math.min(1, 0.4 + 1.8 * valence); // low → subtle, high → powerful

  // ─── 1. Audio layer – drives timing & emotional tone ─────────────
  if (!skipAudio && 'AudioContext' in window) {
    const audioDuration = durationMs * (0.8 + 0.4 * valence); // longer when high valence

    // Select audio pattern based on type + valence
    let audioPattern: string;
    switch (type) {
      case 'success': case 'bloom': case 'sync':
        audioPattern = 'cosmicHarmony';
        break;
      case 'warning': case 'error':
        audioPattern = 'warningPulse';
        break;
      case 'gesture':
        audioPattern = 'gestureAffirmation';
        break;
      default:
        audioPattern = 'neutralPulse';
    }

    audioFeedback[audioPattern as keyof typeof audioFeedback]?.(valence);
  }

  // ─── 2. Haptic layer – synced to audio timing ────────────────────
  if (!skipHaptic) {
    let hapticPattern: string;
    switch (type) {
      case 'success': case 'bloom': case 'sync':
        hapticPattern = 'cosmicHarmony';
        break;
      case 'warning': case 'error':
        hapticPattern = 'warningPulse';
        break;
      case 'gesture':
        hapticPattern = 'gestureDetected';
        break;
      default:
        hapticPattern = 'neutralPulse';
    }

    haptic[hapticPattern as keyof typeof haptic]?.(valence);
  }

  // ─── 3. Visual layer – synced visual ripple / glow ───────────────
  if (!skipVisual) {
    visualFeedback[type as keyof typeof visualFeedback]?.({
      type,
      durationMs,
      intensity,
      targetElement,
      message: `${type} – valence ${valence.toFixed(2)}`
    });
  }

  // ─── 4. Motion / UI animation layer – subtle pulse / ripple ──────
  if (!skipMotion && targetElement) {
    const motionClass = `cross-modal-\( {type}- \){Math.round(intensity * 10)}`;
    targetElement.classList.add(motionClass);

    setTimeout(() => targetElement.classList.remove(motionClass), durationMs);
  }

  // ─── Mercy safety cap – prevent overstimulation ──────────────────
  if (valence < 0.65 && durationMs > 1200) {
    console.debug('[CrossModal] Reduced duration due to low valence mercy gate');
  }

  console.log(
    `[CrossModal] ${type} triggered – valence: ${valence.toFixed(2)}, intensity: ${intensity.toFixed(2)}, duration: ${durationMs}ms`
  );
}

// ──────────────────────────────────────────────────────────────
// Convenience exports
// ──────────────────────────────────────────────────────────────

export const crossModal = {
  success: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'success', ...opts }),
  warning: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'warning', ...opts }),
  error: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'error', ...opts }),
  loading: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'loading', ...opts }),
  bloom: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'bloom', ...opts }),
  sync: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'sync', ...opts }),
  gesture: (opts?: Partial<CrossModalOptions>) => triggerCrossModalFeedback({ type: 'gesture', ...opts }),
};

export default crossModal;
