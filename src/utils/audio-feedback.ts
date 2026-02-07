// src/utils/audio-feedback.ts – Audio Feedback Library v1.0
// Valence-modulated pure tones, chords, sweeps, pulses via Web Audio API
// Mercy-gated volume & frequency, haptic/visual sync, offline-safe (no external files)
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from './haptic-patterns';

let audioContext: AudioContext | null = null;
let masterGain: GainNode | null = null;

// ──────────────────────────────────────────────────────────────
// Core Audio Setup (lazy-init, offline-safe)
// ──────────────────────────────────────────────────────────────

function getAudioContext(): AudioContext {
  if (!audioContext) {
    audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    masterGain = audioContext.createGain();
    masterGain.connect(audioContext.destination);
    masterGain.gain.value = 0.3; // default gentle volume
  }
  return audioContext;
}

// ──────────────────────────────────────────────────────────────
// Valence → Audio Parameters Mapping
// ──────────────────────────────────────────────────────────────

const BASE_FREQ = 432; // A=432 Hz – harmonious base

const AUDIO_PATTERNS = {
  cosmicHarmony: (valence: number) => ({
    type: 'sine',
    freqStart: BASE_FREQ * (1 + 0.2 * valence),
    freqEnd: BASE_FREQ * (1.4 + 0.3 * valence),
    duration: 1800 + 800 * valence,
    volume: 0.25 + 0.15 * valence
  }),

  neutralPulse: (valence: number) => ({
    type: 'triangle',
    freq: BASE_FREQ * 1.2,
    duration: 80 + 40 * valence,
    volume: 0.2 + 0.1 * valence,
    repeat: 2,
    gap: 100
  }),

  warningPulse: (valence: number) => ({
    type: 'sawtooth',
    freq: 220 + 80 * (1 - valence),
    duration: 120,
    volume: 0.35 * (1 - 0.4 * valence), // softer when low valence
    repeat: 3,
    gap: 180
  }),

  bloomBurst: (valence: number) => ({
    type: 'sine',
    freqStart: BASE_FREQ * 0.8,
    freqEnd: BASE_FREQ * 2.5,
    duration: 1200 + 600 * valence,
    volume: 0.4 + 0.2 * valence,
    type: 'sine' as const
  }),

  rejection: (valence: number) => ({
    type: 'sawtooth',
    freq: 180 - 40 * valence,
    duration: 200,
    volume: 0.3 * (1 - valence * 0.5),
    repeat: 2,
    gap: 300
  }),

  criticalAlert: (valence: number) => ({
    type: 'square',
    freq: 300,
    duration: 150,
    volume: 0.45 * (1 - valence * 0.3),
    repeat: 4,
    gap: 200
  })
};

type PatternKey = keyof typeof AUDIO_PATTERNS;

/**
 * Play a valence-modulated audio pattern
 * @param patternKey Pattern name from AUDIO_PATTERNS
 * @param customValence Optional override (defaults to currentValence)
 * @param syncHaptic Whether to sync with haptic pattern (default true)
 */
export async function playAudioPattern(
  patternKey: PatternKey,
  customValence?: number,
  syncHaptic: boolean = true
): Promise<void> {
  const actionName = `Play audio pattern: ${patternKey}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[AudioFeedback] Mercy gate blocked pattern: ${patternKey}`);
    return;
  }

  const valence = customValence ?? currentValence.get();
  const ctx = getAudioContext();
  if (!ctx) return;

  const pattern = AUDIO_PATTERNS[patternKey](valence);
  const { type, freqStart, freqEnd, duration, volume, repeat = 1, gap = 0 } = pattern;

  // Volume mercy cap – never too loud
  masterGain!.gain.value = Math.min(0.4, volume);

  let currentTime = ctx.currentTime;

  for (let i = 0; i < repeat; i++) {
    const osc = ctx.createOscillator();
    osc.type = type as OscillatorType;

    if (freqStart !== undefined && freqEnd !== undefined) {
      // Frequency sweep
      osc.frequency.setValueAtTime(freqStart, currentTime);
      osc.frequency.linearRampToValueAtTime(freqEnd, currentTime + duration / 1000);
    } else if (pattern.freq) {
      osc.frequency.value = pattern.freq;
    }

    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0, currentTime);
    gainNode.gain.linearRampToValueAtTime(volume, currentTime + 0.05);
    gainNode.gain.linearRampToValueAtTime(0, currentTime + duration / 1000 + 0.05);

    osc.connect(gainNode);
    gainNode.connect(masterGain!);

    osc.start(currentTime);
    osc.stop(currentTime + duration / 1000 + 0.1);

    currentTime += (duration + gap) / 1000;
  }

  // Sync haptic if requested
  if (syncHaptic) {
    haptic[patternKey as keyof typeof haptic]?.(valence);
  }

  console.log(`[AudioFeedback] Played ${patternKey} (valence: ${valence.toFixed(2)})`);
}

// ──────────────────────────────────────────────────────────────
// Convenience exports
// ──────────────────────────────────────────────────────────────

export const audioFeedback = {
  cosmicHarmony: (valence?: number) => playAudioPattern('cosmicHarmony', valence),
  neutralPulse: (valence?: number) => playAudioPattern('neutralPulse', valence),
  warningPulse: (valence?: number) => playAudioPattern('warningPulse', valence),
  bloomBurst: (valence?: number) => playAudioPattern('bloomBurst', valence),
  rejection: (valence?: number) => playAudioPattern('rejection', valence),
  criticalAlert: (valence?: number) => playAudioPattern('criticalAlert', valence),
};

export default audioFeedback;
