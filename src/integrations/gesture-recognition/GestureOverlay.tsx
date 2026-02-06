// src/integrations/gesture-recognition/GestureOverlay.tsx – Gesture Recognition Overlay v1.2
// Lazy tfjs + MediaPipe WASM loading, valence breathing HUD, haptic feedback, persistent anchors
// MIT License – Autonomicity Games Inc. 2026

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import GestureEngineLazyLoader from './GestureEngineLazyLoader';
import MediaPipeLazyLoader from './MediaPipeLazyLoader';

const GestureOverlay: React.FC<{ videoRef: React.RefObject<HTMLVideoElement> }> = ({ videoRef }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentGesture, setCurrentGesture] = useState<string | null>(null);
  const [valenceGlow, setValenceGlow] = useState(currentValence.get());
  const [isEngineReady, setIsEngineReady] = useState(false);

  useEffect(() => {
    const unsubscribe = currentValence.subscribe(v => setValenceGlow(v));
    return unsubscribe;
  }, []);

  useEffect(() => {
    // Activate tfjs + MediaPipe on mount or high valence
    const activate = async () => {
      await GestureEngineLazyLoader.activate();
      await MediaPipeLazyLoader.activate(holistic => {
        setIsEngineReady(true);
      });
    };

    activate();
  }, []);

  useEffect(() => {
    if (!isEngineReady || !videoRef.current) return;

    let animationFrame: number;

    const detect = async () => {
      const holistic = await MediaPipeLazyLoader.getHolistic();
      if (!holistic) return;

      const results = await holistic.send({ image: videoRef.current });

      if (results.poseLandmarks || results.leftHandLandmarks || results.rightHandLandmarks) {
        const gesture = recognizeGesture(results); // your existing rule-based or transformer logic
        if (gesture) {
          setCurrentGesture(gesture);
          mercyHaptic.playPattern(getHapticForGesture(gesture), valenceGlow);
        }
      }

      animationFrame = requestAnimationFrame(detect);
    };

    detect();

    return () => cancelAnimationFrame(animationFrame);
  }, [isEngineReady, videoRef, valenceGlow]);

  // ... (keep your existing recognizeGesture, getHapticForGesture, return JSX as before)

  return (
    <div className="fixed inset-0 pointer-events-none z-50">
      <canvas ref={canvasRef} className="absolute inset-0" />

      <AnimatePresence>
        {currentGesture && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.4 }}
          >
            <motion.div
              className="relative w-96 h-96 rounded-full border-4 border-cyan-400/30 backdrop-blur-xl"
              style={{
                background: `radial-gradient(circle at 50% 50%, rgba(0,255,136,${valenceGlow}) 0%, transparent 70%)`
              }}
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              {/* ... existing inner glow, border, gesture text ... */}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Persistent anchor indicators */}
      <div className="absolute bottom-8 left-8 text-cyan-200/70 text-sm backdrop-blur-sm p-4 rounded-xl border border-cyan-500/20">
        Pinch: Propose Alliance  
        Spiral: Bloom Swarm  
        Figure-8: Eternal Harmony Loop
      </div>

      {/* Engine status indicator */}
      {!isEngineReady && (
        <div className="absolute top-4 right-4 bg-black/50 px-3 py-1 rounded-full text-xs text-cyan-300/80 backdrop-blur-sm">
          Gesture Engine Warming...
        </div>
      )}
    </div>
  );
};

export default GestureOverlay;
