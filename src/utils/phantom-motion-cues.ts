// src/utils/phantom-motion-cues.ts – Phantom Motion Cues Engine v1.0
// Creates illusory directional motion via asymmetric vibration, optical flow gradients, parallax hints
// Valence-modulated direction/intensity, mercy-gated for low-power/reduced-motion users
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from './haptic-patterns';

// ──────────────────────────────────────────────────────────────
// Directional Phantom Motion Types
// ──────────────────────────────────────────────────────────────

type MotionDirection = 'left' | 'right' | 'up' | 'down' | 'center' | 'pulse' | 'wave';

interface PhantomMotionOptions {
  direction: MotionDirection;
  intensity?: number;          // 0–1 (valence-scaled)
  durationMs?: number;         // total effect duration
  targetElement?: HTMLElement; // for visual flow/parallax
  syncHaptic?: boolean;        // default true
  syncVisual?: boolean;        // default true
}

/**
 * Trigger phantom motion cue – illusory directional sensation
 * - Asymmetric vibration creates "pull" feeling
 * - Visual flow gradient + subtle parallax enhances illusion
 * - Valence scales intensity & direction sharpness
 */
export async function triggerPhantomMotion(options: PhantomMotionOptions): Promise<void> {
  const { direction, intensity: userIntensity = 1, durationMs = 800, targetElement, syncHaptic = true, syncVisual = true } = options;

  const actionName = `Phantom motion cue: ${direction}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[PhantomMotion] Mercy gate blocked: ${direction}`);
    return;
  }

  const valence = currentValence.get();
  const intensity = Math.min(1, userIntensity * (0.4 + 1.6 * valence)); // low → subtle, high → vivid

  // ─── 1. Asymmetric Haptic Pattern for Phantom Pull ──────────────
  if (syncHaptic && 'vibrate' in navigator) {
    let hapticPattern: number[] = [];

    switch (direction) {
      case 'left':
        hapticPattern = [120 * intensity, 40, 40 * intensity, 80, 80 * intensity];
        break;
      case 'right':
        hapticPattern = [40 * intensity, 80, 80 * intensity, 40, 120 * intensity];
        break;
      case 'up':
        hapticPattern = [100 * intensity, 60, 60 * intensity, 100];
        break;
      case 'down':
        hapticPattern = [60 * intensity, 100, 100 * intensity, 60];
        break;
      case 'center':
        hapticPattern = [80 * intensity, 50, 100 * intensity, 50, 80 * intensity];
        break;
      case 'pulse':
        hapticPattern = [60 * intensity, 120, 80 * intensity, 100, 60 * intensity];
        break;
      case 'wave':
        hapticPattern = [40, 80, 60, 100, 80, 80, 60, 100];
        break;
    }

    navigator.vibrate(hapticPattern);
  }

  // ─── 2. Visual Flow Gradient + Parallax Hint ─────────────────────
  if (syncVisual && targetElement) {
    const flowEl = document.createElement('div');
    flowEl.className = 'phantom-flow';
    flowEl.style.position = 'absolute';
    flowEl.style.inset = '0';
    flowEl.style.pointerEvents = 'none';
    flowEl.style.background = `linear-gradient(to \( {direction}, transparent, rgba(0,255,136, \){0.2 * intensity}), transparent)`;
    flowEl.style.backgroundSize = '200% 200%';
    flowEl.style.animation = `phantom-flow-${direction} ${durationMs / 1000}s ease-out forwards`;
    targetElement.style.position = 'relative';
    targetElement.appendChild(flowEl);

    // Subtle parallax hint (tilt illusion)
    const tiltX = direction === 'left' ? -5 : direction === 'right' ? 5 : 0;
    const tiltY = direction === 'up' ? -5 : direction === 'down' ? 5 : 0;
    targetElement.style.transform = `perspective(1000px) rotateX(\( {tiltY}deg) rotateY( \){tiltX}deg)`;
    setTimeout(() => {
      targetElement.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
      flowEl.remove();
    }, durationMs);
  }

  // ─── 3. Mercy power envelope – prevent overstimulation ──────────
  if (valence < 0.65 && durationMs > 1200) {
    console.debug('[PhantomMotion] Reduced duration due to low valence mercy gate');
  }

  console.log(
    `[PhantomMotion] ${direction} cue triggered – valence: ${valence.toFixed(2)}, intensity: ${intensity.toFixed(2)}, duration: ${durationMs}ms`
  );
}

// CSS animations (inject once)
if (!document.getElementById('phantom-motion-css')) {
  const style = document.createElement('style');
  style.id = 'phantom-motion-css';
  style.textContent = `
    @keyframes phantom-flow-left  { 0% { background-position: 0% 50%; } 100% { background-position: -200% 50%; } }
    @keyframes phantom-flow-right { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    @keyframes phantom-flow-up    { 0% { background-position: 50% 0%; } 100% { background-position: 50% -200%; } }
    @keyframes phantom-flow-down  { 0% { background-position: 50% 0%; } 100% { background-position: 50% 200%; } }
    @keyframes phantom-flow-center { 0% { opacity: 0.6; } 100% { opacity: 0; } }
  `;
  document.head.appendChild(style);
}

// ──────────────────────────────────────────────────────────────
// Convenience exports
// ──────────────────────────────────────────────────────────────

export const phantomMotion = {
  left: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'left', ...opts }),
  right: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'right', ...opts }),
  up: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'up', ...opts }),
  down: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'down', ...opts }),
  center: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'center', ...opts }),
  pulse: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'pulse', ...opts }),
  wave: (opts?: Partial<PhantomMotionOptions>) => triggerPhantomMotion({ direction: 'wave', ...opts }),
};

export default phantomMotion;
