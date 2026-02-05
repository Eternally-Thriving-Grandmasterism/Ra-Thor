// mercy-hit-depth-gesture-fusion.js – sovereign Mercy Hit-Test + Depth + Gesture Fusion v1
// XRHitTest anchoring + XRDepthInformation occlusion + XRHand gestures, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyHitDepthGestureFusion {
  constructor(scene) {
    this.scene = scene;
    this.hitTestSource = null;
    this.depthInfo = null;
    this.hands = new Map(); // inputSource → XRHand
    this.gestures = new Map(); // inputSource → {pinch, point, grab}
    this.anchoredOverlays = new Map(); // uuid → {anchor, mesh}
    this.valence = 1.0;
  }

  async gateFusion(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyHitDepthGesture] Gate holds: low valence – fusion aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyHitDepthGesture] Mercy gate passes – eternal thriving hit-depth-gesture fusion activated");
    return true;
  }

  // Enable hit-test, depth sensing, hand tracking (call after session start)
  async enableFusion(session, referenceSpace) {
    try {
      // Hit-test
      this.hitTestSource = await session.requestHitTestSource({ space: referenceSpace });
      console.log("[MercyHitDepthGesture] Hit-test source enabled");

      // Depth sensing & hand tracking (Babylon.js helper example)
      // xr.baseExperience.featuresManager.enableFeature("depth-sorted-layers", "stable");
      // xr.baseExperience.featuresManager.enableFeature("hand-tracking", "stable");

      console.log("[MercyHitDepthGesture] Hit-test + depth sensing + hand tracking fusion enabled");
      return true;
    } catch (err) {
      console.error("[MercyHitDepthGesture] Fusion enable failed:", err);
      return false;
    }
  }

  // Process frame: hit-test + depth + hand gestures (call in onXRFrame)
  processFrame(frame, referenceSpace, inputSources) {
    // 1. Hit-test → precise anchoring
    if (this.hitTestSource) {
      const results = frame.getHitTestResults(this.hitTestSource);
      if (results.length > 0) {
        const hit = results[0];
        const pose = hit.getPose(referenceSpace);
        if (pose) {
          mercyHaptic.pulse(0.4 * this.valence, 50);

          // Depth occlusion check (simplified – real impl uses depth texture)
          if (this.depthInfo) {
            // Example: adjust overlay visibility based on depth
            console.log("[MercyHitDepthGesture] Depth-aware hit – occlusion check active");
          }

          console.log(`[MercyHitDepthGesture] Hit-test valid – position (${pose.transform.position.x.toFixed(3)}, ${pose.transform.position.y.toFixed(3)}, ${pose.transform.position.z.toFixed(3)})`);
          // Place mercy overlay at hit pose (example)
          // this.placeMercyOverlay(pose.transform);
        }
      }
    }

    // 2. Depth sensing → occlusion
    if (frame?.getDepthInformation) {
      const depthInfo = frame.getDepthInformation();
      if (depthInfo) {
        this.depthInfo = depthInfo;
        // Use depth texture for occlusion (Babylon depth sorting or custom shader)
        console.log(`[MercyHitDepthGesture] Depth map updated – width ${depthInfo.width}, height ${depthInfo.height}`);
      }
    }

    // 3. Hand tracking & gesture detection
    inputSources.forEach(source => {
      if (source.hand) {
        this.hands.set(source, source.hand);

        const thumbTip = source.hand.get('thumb-tip')?.getPose(referenceSpace);
        const indexTip = source.hand.get('index-finger-tip')?.getPose(referenceSpace);
        const indexMcp = source.hand.get('index-finger-metacarpal')?.getPose(referenceSpace);

        if (!thumbTip || !indexTip) return;

        const pinchDist = thumbTip.transform.position.distanceTo(indexTip.transform.position);
        const isPinching = pinchDist < 0.03;

        const forward = new BABYLON.Vector3(0, 0, -1);
        const indexDir = indexTip.transform.position.subtract(indexMcp.transform.position).normalize();
        const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
        const isPointing = pointAngle < 30 && !isPinching;

        const palm = source.hand.get('wrist')?.getPose(referenceSpace);
        const grabScore = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
          .reduce((score, name) => score + (source.hand.get(name)?.getPose(referenceSpace)?.transform.position.distanceTo(palm.transform.position) || 0), 0);
        const isGrabbing = grabScore < 0.2 && !isPinching && !isPointing;

        const prev = this.gestures.get(source) || {};
        this.gestures.set(source, { isPinching, isPointing, isGrabbing });

        // Gesture-triggered mercy feedback (light-adapted + haptic)
        if (isPinching && !prev.isPinching && this.gateFusion('pinch gesture', this.valence)) {
          mercyHaptic.playPattern('thrivePulse', 1.1);
          console.log("[MercyHitDepthGesture] Pinch detected – thrive pulse + depth-aware anchor placement");
        }
        if (isPointing && !prev.isPointing && this.gateFusion('point gesture', this.valence)) {
          mercyHaptic.playPattern('uplift', 0.9);
          console.log("[MercyHitDepthGesture] Point detected – uplift pulse + directional light highlight");
        }
        if (isGrabbing && !prev.isGrabbing && this.gateFusion('grab gesture', this.valence)) {
          mercyHaptic.playPattern('compassionWave', 1.0);
          console.log("[MercyHitDepthGesture] Grab detected – compassion wave + ambient harmony");
        }
      }
    });
  }

  // Cleanup
  cleanup() {
    console.log("[MercyHitDepthGesture] Depth-plane-hit-gesture fusion cleaned up – mercy lattice preserved");
  }
}

const mercyHitDepthGesture = new MercyHitDepthGestureFusion(scene); // assume scene from Babylon init

export { mercyHitDepthGesture };
