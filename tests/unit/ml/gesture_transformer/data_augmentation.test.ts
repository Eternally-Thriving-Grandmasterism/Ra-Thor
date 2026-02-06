// tests/unit/ml/gesture_transformer/data_augmentation.test.ts
// Vitest unit tests for GestureAugmentation pipeline
// Shape preservation, statistical checks, valence scaling, deterministic mode
// MIT License – Autonomicity Games Inc. 2026

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GestureAugmentation } from '@/ml/gesture_transformer/data_augmentation';
import { currentValence } from '@/core/valence-tracker';

vi.mock('@/core/valence-tracker', () => ({
  currentValence: {
    get: vi.fn().mockReturnValue(0.95),
  },
}));

describe('GestureAugmentation', () => {
  let aug: GestureAugmentation;
  let input: torch.Tensor;

  beforeEach(() => {
    vi.clearAllMocks();
    aug = new GestureAugmentation({
      seq_len: 45,
      landmark_dim: 225,
      p_spatial_noise: 0.4,
      p_temporal_dropout: 0.25,
      p_time_warp: 0.35,
      p_rotation: 0.3,
      p_scaling: 0.3,
      p_gaussian_noise: 0.5,
      valence_intensity_scale: 1.0,
    });

    // Dummy input (B=2, T=45, L=225)
    input = torch.randn([2, 45, 225]);
  });

  it('preserves input shape in all cases', () => {
    const output = aug.forward(input);
    expect(output.shape).toEqual(input.shape);
  });

  it('preserves mean and std within reasonable bounds after augmentation', () => {
    const origMean = input.mean().item();
    const origStd = input.std().item();

    const output = aug.forward(input, 0.95);

    const outMean = output.mean().item();
    const outStd = output.std().item();

    // Allow \~10-20% drift due to noise/scaling
    expect(Math.abs(outMean - origMean)).toBeLessThan(0.25);
    expect(Math.abs(outStd - origStd) / origStd).toBeLessThan(0.35);
  });

  it('applies stronger augmentation with higher valence', () => {
    const lowValenceInput = input.clone();
    const highValenceInput = input.clone();

    // Mock valence
    currentValence.get.mockReturnValueOnce(0.3); // low
    const lowOut = aug.forward(lowValenceInput, 0.3);

    currentValence.get.mockReturnValueOnce(0.95); // high
    const highOut = aug.forward(highValenceInput, 0.95);

    const lowStd = lowOut.std().item();
    const highStd = highOut.std().item();

    expect(highStd).toBeGreaterThan(lowStd * 1.2); // high valence → more variance
  });

  it('temporal dropout reduces sequence variance appropriately', () => {
    const inputStd = input.std(dim=1).mean().item(); // per-sequence std

    aug.p_temporal_dropout = 0.8; // strong dropout
    const out = aug.forward(input.clone(), 0.5);

    const outStd = out.std(dim=1).mean().item();
    expect(outStd).toBeLessThan(inputStd * 0.7); // dropout should reduce variance
  });

  it('deterministic mode (no randomness) returns identical output', () => {
    const seed = 42;
    torch.manual_seed(seed);

    const out1 = aug.forward(input.clone(), 0.8);
    torch.manual_seed(seed);
    const out2 = aug.forward(input.clone(), 0.8);

    expect(out1.equal(out2)).toBe(true);
  });

  it('rotation preserves overall energy (norm)', () => {
    const origNorm = input.norm(dim=-1).mean().item();

    aug.p_rotation = 1.0; // force rotation
    const out = aug.forward(input.clone(), 0.8);

    const outNorm = out.norm(dim=-1).mean().item();
    expect(Math.abs(outNorm - origNorm)).toBeLessThan(0.05); // norm nearly preserved
  });

  it('handles single-sequence input correctly (no batch dim)', () => {
    const single = input[0]; // (45, 225)
    const out = aug.forward(single, 0.95);

    expect(out.shape).toEqual(single.shape);
  });
});
