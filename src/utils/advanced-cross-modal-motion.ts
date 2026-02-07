// src/utils/advanced-cross-modal-motion.ts – Advanced Cross-Modal Motion Integration v1.0
// Spring-physics based motion, micro-oscillations, directional phantom cues
// Valence-modulated amplitude/frequency/direction, mercy power envelope
// Syncs with haptic/visual/audio for unified multisensory experience
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from './haptic-patterns';
import visualFeedback from './visual-feedback';
import audioFeedback from './audio-feedback';

// ──────────────────────────────────────────────────────────────
// Spring Physics Simulation (for smooth, natural-feeling motion)
// ──────────────────────────────────────────────────────────────

interface SpringParams {
  stiffness: number;    // 50–300 (higher = snappier)
  damping: number;      // 0.5–1.2 (higher = less oscillation)
  mass: number;         // 0.8–2.0
  precision?: number;   // animation frame precision
}

function springAnimation(
  target: HTMLElement,
  property: 'scale' | 'translateX' | 'translateY' | 'rotate',
  toValue: number,
  params: SpringParams = { stiffness: 180, damping: 12, mass: 1 },
  durationMs: number = 800
): Promise<void> {
  return new Promise(resolve => {
    const startTime = performance.now();
    const startValue = parseFloat(getComputedStyle(target)[property as any]) || 1;

    const animate = (time: number) => {
      const elapsed = time - startTime;
      const t = elapsed / durationMs;

      if (t >= 1) {
        target.style.transform = `\( {property}( \){toValue})`;
        resolve();
        return;
      }

      // Spring equation (simple damped harmonic oscillator)
      const velocity = 0;
      const displacement = toValue - startValue;
      const dampingRatio = params.damping / (2 * Math.sqrt(params.stiffness * params.mass));
      const omega = Math.sqrt(params.stiffness / params.mass);
      const decay = Math.exp(-dampingRatio * omega * t);
      const oscillation = Math.cos(omega * Math.sqrt(1 - dampingRatio ** 2) * t);

      const value = startValue + displacement * (1 - decay * oscillation);

      target.style.transform = `\( {property}( \){value})`;

      requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  });
}

// ──────────────────────────────────────────────────────────────
// Phantom Motion Cues (directional vibration + visual flow)
// ──────────────────────────────────────────────────────────────

function triggerPhantomMotion(
  direction: 'left' | 'right' | 'up' | 'down' | 'center',
  intensity: number = 1,
  durationMs: number = 600
) {
  const valence = currentValence.get();
  const safeIntensity = Math.min(1, intensity * (0.5 + 1.5 * valence));

  // Haptic directional pattern (simulated via asymmetric pulses)
  const pattern = direction === 'center'
    ? [80 * safeIntensity, 40, 80 * safeIntensity]
    : direction === 'left'
    ? [120 * safeIntensity, 60, 40 * safeIntensity]
    : direction === 'right'
    ? [40 * safeIntensity, 60, 120 * safeIntensity]
    : direction === 'up'
    ? [100 * safeIntensity, 80, 60 * safeIntensity]
    : [60 * safeIntensity, 80, 100 * safeIntensity];

  navigator.vibrate?.(pattern);

  // Visual flow cue (subtle gradient shift or particle drift)
  const flowEl = document.createElement('div');
  flowEl.style.position = 'fixed';
  flowEl.style.inset = '0';
  flowEl.style.pointerEvents = 'none';
  flowEl.style.background = `linear-gradient(to \( {direction}, transparent, rgba(0,255,136, \){0.15 * safeIntensity}), transparent)`;
  flowEl.style.animation = `flow-${direction} ${durationMs / 1000}s ease-out forwards`;
  document.body.appendChild(flowEl);

  setTimeout(() => flowEl.remove(), durationMs);
}

// CSS for flow animation (inject once)
if (!document.getElementById('phantom-motion-style')) {
  const style = document.createElement('style');
  style.id = 'phantom-motion-style';
  style.textContent = `
    @keyframes flow-left  { to { background-position: -100% 50%; } }
    @keyframes flow-right { to { background-position: 100% 50%; } }
    @keyframes flow-up    { to { background-position: 50% -100%; } }
    @keyframes flow-down  { to { background-position: 50% 100%; } }
    @keyframes flow-center { to { opacity: 0; } }
  `;
  document.head.appendChild(style);
}

// ──────────────────────────────────────────────────────────────
// Unified Cross-Modal Motion Trigger
// ──────────────────────────────────────────────────────────────

export async function triggerAdvancedMotion(
  type: 'success' | 'warning' | 'bloom' | 'gesture' | 'sync' | 'alert',
  targetElement?: HTMLElement,
  valenceOverride?: number
): Promise<void> {
  const actionName = `Advanced cross-modal motion: ${type}`;
  if (!await mercyGate(actionName)) return;

  const valence = valenceOverride ?? currentValence.get();
  const intensity = 0.4 + 1.6 * valence;

  // ─── Haptic base rhythm ────────────────────────────────────────
  let hapticPattern: number[] = [];
  switch (type) {
    case 'success': case 'bloom': case 'sync':
      hapticPattern = [60, 40, 80, 30, 100, 20, 120, 50];
      break;
    case 'warning': case 'alert':
      hapticPattern = [120, 80, 80, 100, 120];
      break;
    case 'gesture':
      hapticPattern = [80, 40, 100, 30, 120];
      break;
  }
  hapticPattern = hapticPattern.map(v => Math.round(v * intensity));
  navigator.vibrate?.(hapticPattern);

  // ─── Visual motion (spring + ripple) ───────────────────────────
  if (targetElement) {
    // Spring pulse on target
    springAnimation(targetElement, 'scale', 1 + 0.15 * intensity, {
      stiffness: 220 + 80 * valence,
      damping: 10 + 4 * valence,
      mass: 1
    }, 800);

    // Ripple from center
    visualFeedback[type as keyof typeof visualFeedback]?.({
      type,
      durationMs: 1200,
      intensity,
      targetElement
    });
  }

  // ─── Phantom directional cue (optional) ────────────────────────
  if (type === 'gesture' || type === 'bloom') {
    triggerPhantomMotion('center', intensity, 800);
  }

  // ─── Audio sync pulse (subtle tone) ────────────────────────────
  if ('AudioContext' in window) {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = 220 + 180 * valence;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.15 * intensity, ctx.currentTime + 0.05);
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.4);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.45);
  }

  console.log(`[AdvancedMotion] ${type} triggered – valence: ${valence.toFixed(2)}, intensity: ${intensity.toFixed(2)}`);
}

export default triggerAdvancedMotion;
