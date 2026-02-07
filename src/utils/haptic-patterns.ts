// src/utils/haptic-patterns.ts – Haptic Patterns Library v1.0
// Valence-modulated vibration sequences for cosmic harmony, warning, bloom, etc.
// Mercy-gated, progressive intensity, cross-platform (navigator.vibrate)
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

// Pattern definitions – [duration_ms, pause_ms, duration_ms, ...]
const PATTERNS = {
  // Cosmic Harmony – gentle rising wave for success, high valence
  cosmicHarmony: (intensity: number = 1) => [
    60 * intensity, 40,
    80 * intensity, 30,
    100 * intensity, 20,
    120 * intensity, 50,
    80 * intensity
  ],

  // Neutral Pulse – calm confirmation
  neutralPulse: (intensity: number = 1) => [
    40 * intensity, 60,
    40 * intensity
  ],

  // Warning Pulse – urgent but non-alarming (mercy gate trigger)
  warningPulse: (intensity: number = 1) => [
    120 * intensity, 80,
    80 * intensity, 100,
    120 * intensity
  ],

  // Bloom Burst – celebratory rapid pulses for growth/emergence
  bloomBurst: (intensity: number = 1) => [
    30 * intensity, 20,
    40 * intensity, 15,
    50 * intensity, 10,
    60 * intensity, 80,
    40 * intensity
  ],

  // Rejection / Low Valence – short, sharp, diminishing
  rejection: (intensity: number = 1) => [
    80 * intensity, 120,
    60 * intensity, 140,
    40 * intensity
  ],

  // Critical Alert – strong but short (system-level mercy block)
  criticalAlert: (intensity: number = 1) => [
    200 * intensity, 100,
    150 * intensity
  ]
};

type PatternKey = keyof typeof PATTERNS;

/**
 * Play a haptic pattern with valence-modulated intensity
 * @param patternKey Pattern name from PATTERNS
 * @param customValence Optional override (defaults to currentValence)
 * @param maxDurationMs Optional cap (default unlimited)
 */
export async function playPattern(
  patternKey: PatternKey,
  customValence?: number,
  maxDurationMs: number = 2000
): Promise<void> {
  const actionName = `Play haptic pattern: ${patternKey}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[Haptics] Mercy gate blocked pattern: ${patternKey}`);
    return;
  }

  if (!('vibrate' in navigator)) {
    console.debug('[Haptics] Vibration API not supported');
    return;
  }

  const valence = customValence ?? currentValence.get();
  // Intensity scale: 0.3 → 1.0 (low valence = gentle, high = powerful)
  const intensity = 0.3 + 0.7 * valence;
  const pattern = PATTERNS[patternKey](intensity);

  // Cap total duration for mercy
  let totalDuration = 0;
  const cappedPattern: number[] = [];
  for (const duration of pattern) {
    if (totalDuration + duration > maxDurationMs) break;
    cappedPattern.push(duration);
    totalDuration += duration;
  }

  try {
    navigator.vibrate(cappedPattern);
    console.log(`[Haptics] Played ${patternKey} (valence: ${valence.toFixed(2)}, intensity: ${intensity.toFixed(2)})`);
  } catch (err) {
    console.warn('[Haptics] Vibration failed:', err);
  }
}

/**
 * Convenience wrappers
 */
export const haptic = {
  cosmicHarmony: (valence?: number) => playPattern('cosmicHarmony', valence),
  neutralPulse: (valence?: number) => playPattern('neutralPulse', valence),
  warningPulse: (valence?: number) => playPattern('warningPulse', valence),
  bloomBurst: (valence?: number) => playPattern('bloomBurst', valence),
  rejection: (valence?: number) => playPattern('rejection', valence),
  criticalAlert: (valence?: number) => playPattern('criticalAlert', valence),
};

export default haptic;
