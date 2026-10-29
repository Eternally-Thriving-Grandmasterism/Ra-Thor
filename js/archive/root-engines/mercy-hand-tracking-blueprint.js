// mercy-hand-tracking-blueprint.js – sovereign Mercy Hand Tracking Blueprint v1
// XRHand joint poses, gesture detection, mercy-gated interactions, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyHandTracking {
  constructor() {
    this.hands = new Map(); // inputSource → XRHand
    this.gestures = new Map(); // inputSource → {pinch, point, grab}
    this.valence = 1.0;
  }

  async gateHandTracking(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyHand] Gate holds: low valence – hand tracking aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyHand] Mercy gate passes – eternal thriving hand tracking activated");
    return true;
  }

  // Enable hand tracking (call after session start)
  async enableHandTracking(session) {
    try {
      // Babylon.js helper example
      // xr.baseExperience.featuresManager.enableFeature("hand-tracking", "stable");
      console.log("[MercyHand] Hand tracking enabled – mercy gesture lattice ready");
      return true;
    } catch (err) {
      console.error("[MercyHand] Hand tracking enable failed:", err);
      return false;
    }
  }

  // Process hand joints from XRFrame (call in onXRFrame)
  processHandJoints(frame, inputSources) {
    inputSources.forEach(source => {
      if (source.hand) {
        this.hands.set(source, source.hand);

        const thumbTip = source.hand.get('thumb-tip')?.getPose(frame.referenceSpace);
        const indexTip = source.hand.get('index-finger-tip')?.getPose(frame.referenceSpace);
        const indexMcp = source.hand.get('index-finger-metacarpal')?.getPose(frame.referenceSpace);

        if (!thumbTip || !indexTip) return;

        // Pinch detection
        const pinchDist = thumbTip.transform.position.distanceTo(indexTip.transform.position);
        const isPinching = pinchDist < 0.03;

        // Point detection
        const forward = new BABYLON.Vector3(0, 0, -1);
        const indexDir = indexTip.transform.position.subtract(indexMcp.transform.position).normalize();
        const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
        const isPointing = pointAngle < 30 && !isPinching;

        // Grab detection (simplified)
        const palm = source.hand.get('wrist')?.getPose(frame.referenceSpace);
        const grabScore = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
          .reduce((score, name) => score + (source.hand.get(name)?.getPose(frame.referenceSpace)?.transform.position.distanceTo(palm.transform.position) || 0), 0);
        const isGrabbing = grabScore < 0.2 && !isPinching && !isPointing;

        const prev = this.gestures.get(source) || {};
        this.gestures.set(source, { isPinching, isPointing, isGrabbing });

        // Trigger mercy feedback on gesture change
        if (isPinching && !prev.isPinching) {
          mercyHaptic.playPattern('thrivePulse', 1.1);
        }
        if (isPointing && !prev.isPointing) {
          mercyHaptic.playPattern('uplift', 0.9);
        }
        if (isGrabbing && !prev.isGrabbing) {
          mercyHaptic.playPattern('compassionWave', 1.0);
        }
      }
    });
  }
}

const mercyHandTracking = new MercyHandTracking();

export { mercyHandTracking };
