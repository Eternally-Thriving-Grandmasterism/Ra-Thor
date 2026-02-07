// src/core/accessibility-compliance-layer.ts – Accessibility Compliance Layer v1.0
// WCAG 2.2 AA+ enforcement, screen reader support, keyboard navigation, reduced motion respect
// Valence-aware dynamic adjustments, mercy-gated animation intensity
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

// ──────────────────────────────────────────────────────────────
// Media Query Helpers – Respect user preferences
// ──────────────────────────────────────────────────────────────

export const prefersReducedMotion = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export const prefersHighContrast = () =>
  window.matchMedia('(prefers-contrast: more)').matches ||
  window.matchMedia('(forced-colors: active)').matches;

export const prefersDarkMode = () =>
  window.matchMedia('(prefers-color-scheme: dark)').matches;

// ──────────────────────────────────────────────────────────────
// Mercy-Gated Animation Intensity
// ──────────────────────────────────────────────────────────────

export function getAnimationIntensity(base: number = 1): number {
  const valence = currentValence.get();

  if (prefersReducedMotion()) {
    return 0; // no motion
  }

  // High valence → more joyful movement (when allowed)
  // Low valence → subtle / grounding
  return mercyGate('Animation intensity adjustment')
    ? base * (0.3 + 1.4 * valence)
    : base * 0.4; // very conservative when gate blocks
}

// ──────────────────────────────────────────────────────────────
// ARIA Live Region Helper
// ──────────────────────────────────────────────────────────────

let ariaLiveRegion: HTMLElement | null = null;

export function announce(message: string, politeness: 'polite' | 'assertive' = 'polite') {
  if (!ariaLiveRegion) {
    ariaLiveRegion = document.createElement('div');
    ariaLiveRegion.setAttribute('aria-live', politeness);
    ariaLiveRegion.setAttribute('role', 'status');
    ariaLiveRegion.style.position = 'absolute';
    ariaLiveRegion.style.left = '-9999px';
    ariaLiveRegion.style.width = '1px';
    ariaLiveRegion.style.height = '1px';
    ariaLiveRegion.style.overflow = 'hidden';
    document.body.appendChild(ariaLiveRegion);
  }

  ariaLiveRegion.textContent = message;
  console.debug(`[ARIA Live] ${politeness.toUpperCase()}: ${message}`);
}

// ──────────────────────────────────────────────────────────────
// Focus Management Helpers
// ──────────────────────────────────────────────────────────────

export function trapFocus(element: HTMLElement) {
  const focusable = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusable[0] as HTMLElement;
  const last = focusable[focusable.length - 1] as HTMLElement;

  const handler = (e: KeyboardEvent) => {
    if (e.key === 'Tab') {
      if (e.shiftKey) {
        if (document.activeElement === first) {
          last.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === last) {
          first.focus();
          e.preventDefault();
        }
      }
    }
  };

  element.addEventListener('keydown', handler);
  return () => element.removeEventListener('keydown', handler);
}

// ──────────────────────────────────────────────────────────────
// High-Contrast & Reduced-Motion Classes
// ──────────────────────────────────────────────────────────────

export function applyUserPreferences() {
  document.documentElement.classList.toggle('reduced-motion', prefersReducedMotion());
  document.documentElement.classList.toggle('high-contrast', prefersHighContrast());
  document.documentElement.classList.toggle('dark-mode', prefersDarkMode());
}

// Initialize on load & preference change
if (typeof window !== 'undefined') {
  applyUserPreferences();

  const mqMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const mqContrast = window.matchMedia('(prefers-contrast: more)');
  const mqDark = window.matchMedia('(prefers-color-scheme: dark)');

  mqMotion.addEventListener('change', applyUserPreferences);
  mqContrast.addEventListener('change', applyUserPreferences);
  mqDark.addEventListener('change', applyUserPreferences);
}

// ──────────────────────────────────────────────────────────────
// Global accessibility helpers
// ──────────────────────────────────────────────────────────────

export const a11y = {
  announce,
  trapFocus,
  getAnimationIntensity,
  applyUserPreferences,
};

export default a11y;
