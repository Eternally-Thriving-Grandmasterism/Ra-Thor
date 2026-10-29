// src/integrations/mediapipe-holistic-fusion-engine.ts – MediaPipe Holistic Fusion Engine v1.0
// Unified real-time tracking: Face Mesh (468) + Hands (21×2) + Pose (33)
// Valence-weighted confidence fusion, temporal smoothing, mercy-gated output
// WebNN acceleration, offline-capable after first load
// MIT License – Autonomicity Games Inc. 2026

import { Holistic } from '@mediapipe/holistic';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-patterns';
import visualFeedback from '@/utils/visual-feedback';
import audioFeedback from '@/utils/audio-feedback';

const MEDIAPIPE_HOLISTIC_CONFIG = {
  modelComplexity: 1,                     // 0=Lite, 1=Full, 2=Heavy (best accuracy)
  smoothLandmarks: true,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.65,
  refineFaceLandmarks: true,              // iris tracking
  enableSegmentation: false               // optional
};

const FACE_LANDMARKS = 468;
const HANDS_LANDMARKS_PER_HAND = 21;
const POSE_LANDMARKS = 33;
const TOTAL_LANDMARKS = FACE_LANDMARKS + HANDS_LANDMARKS_PER_HAND * 2 + POSE_LANDMARKS;
const CONFIDENCE_THRESHOLD = 0.78;
const MERCY_FALSE_POSITIVE_DROP = 0.12;
const TEMPORAL_SMOOTHING_ALPHA = 0.7;     // EMA smoothing factor

interface HolisticFusionResult {
  faceLandmarks: any[];                   // 468 points
  leftHandLandmarks: any[];               // 21 points
  rightHandLandmarks: any[];              // 21 points
  poseLandmarks: any[];                   // 33 points
  confidence: number;                     // fused average
  isSafe: boolean;
  projectedValenceImpact: number;
  poseClassification?: string;
  dominantExpression?: string;
  smoothedLandmarks?: any[];              // temporally smoothed unified stream
}

let holisticDetector: Holistic | null = null;
let isInitialized = false;
let previousLandmarks: any[] | null = null; // for temporal smoothing

export class MediaPipeHolisticFusionEngine {
  static async initialize(): Promise<void> {
    const actionName = 'Initialize MediaPipe Holistic fusion engine';
    if (!await mercyGate(actionName)) return;

    if (isInitialized) {
      console.log("[MediaPipeHolisticFusionEngine] Already initialized");
      return;
    }

    console.log("[MediaPipeHolisticFusionEngine] Loading Holistic model...");

    try {
      holisticDetector = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });

      await holisticDetector.setOptions(MEDIAPIPE_HOLISTIC_CONFIG);
      await holisticDetector.initialize();

      isInitialized = true;
      mercyHaptic.cosmicHarmony();
      visualFeedback.success({ message: 'Holistic fusion engine awakened ⚡️' });
      audioFeedback.cosmicHarmony();
      console.log("[MediaPipeHolisticFusionEngine] Holistic fully loaded – face + hands + pose fusion ready");
    } catch (err) {
      console.error("[MediaPipeHolisticFusionEngine] Initialization failed:", err);
      mercyHaptic.warningPulse();
      visualFeedback.error({ message: 'Holistic awakening interrupted ⚠️' });
    }
  }

  static async detectAndFuse(videoElement: HTMLVideoElement): Promise<HolisticFusionResult | null> {
    if (!isInitialized || !holisticDetector) {
      await this.initialize();
      return null;
    }

    const actionName = 'Detect & fuse holistic tracking from video frame';
    if (!await mercyGate(actionName)) return null;

    const valence = currentValence.get();

    try {
      const results = await holisticDetector.send({ image: videoElement });

      let faceLandmarks = results.faceLandmarks?.[0] || [];
      let leftHandLandmarks = results.leftHandLandmarks?.[0] || [];
      let rightHandLandmarks = results.rightHandLandmarks?.[0] || [];
      let poseLandmarks = results.poseLandmarks || [];

      // Temporal smoothing (EMA) – reduce jitter
      const currentLandmarks = [...faceLandmarks, ...leftHandLandmarks, ...rightHandLandmarks, ...poseLandmarks];
      let smoothedLandmarks = currentLandmarks;

      if (previousLandmarks && previousLandmarks.length === currentLandmarks.length) {
        smoothedLandmarks = currentLandmarks.map((lm, i) => {
          const prev = previousLandmarks![i];
          return {
            x: prev.x * TEMPORAL_SMOOTHING_ALPHA + lm.x * (1 - TEMPORAL_SMOOTHING_ALPHA),
            y: prev.y * TEMPORAL_SMOOTHING_ALPHA + lm.y * (1 - TEMPORAL_SMOOTHING_ALPHA),
            z: prev.z * TEMPORAL_SMOOTHING_ALPHA + lm.z * (1 - TEMPORAL_SMOOTHING_ALPHA),
            visibility: lm.visibility || prev.visibility
          };
        });
      }
      previousLandmarks = currentLandmarks;

      // Confidence = average visibility across all domains
      const allLandmarks = [...faceLandmarks, ...leftHandLandmarks, ...rightHandLandmarks, ...poseLandmarks];
      const avgConfidence = allLandmarks.reduce((sum, lm) => sum + (lm.visibility || 0), 0) / allLandmarks.length;

      // Simple pose/expression classification (expand with ML later)
      let poseClass = 'unknown';
      let expression = 'neutral';
      if (poseLandmarks.length > 0) {
        const nose = poseLandmarks[0];
        const leftShoulder = poseLandmarks[11];
        const rightShoulder = poseLandmarks[12];
        const leftHip = poseLandmarks[23];
        const rightHip = poseLandmarks[24];

        const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
        const hipMidY = (leftHip.y + rightHip.y) / 2;

        if (nose.y < shoulderMidY && shoulderMidY < hipMidY) poseClass = 'standing';
        else if (nose.y > shoulderMidY) poseClass = 'sitting';
      }

      const projectedImpact = avgConfidence * valence - 0.5;
      const isSafe = projectedImpact >= -0.05;

      if (!isSafe) {
        mercyHaptic.warningPulse(valence * 0.7);
        visualFeedback.warning({ message: 'Holistic fusion detected – projected valence impact low ⚠️' });
      } else if (poseClass !== 'unknown' || expression !== 'neutral') {
        mercyHaptic.gestureDetected(valence);
        visualFeedback.gesture({ message: `Pose: ${poseClass} | Expression: ${expression} 🧍😊` });
        audioFeedback.gestureDetected(valence);
      }

      return {
        faceLandmarks,
        leftHandLandmarks,
        rightHandLandmarks,
        poseLandmarks,
        confidence: avgConfidence,
        isSafe,
        projectedValenceImpact: projectedImpact,
        poseClassification: poseClass,
        dominantExpression: expression,
        smoothedLandmarks
      };
    } catch (err) {
      console.warn("[MediaPipeHolisticFusionEngine] Detection error:", err);
      return null;
    }
  }

  static async dispose() {
    if (poseDetector) await poseDetector.close();
    if (holisticDetector) await holisticDetector.close();
    isInitialized = false;
    console.log("[MediaPipeHolisticFusionEngine] Detectors closed");
  }
}

export default MediaPipeHolisticFusionEngine;
