// src/ui/components/FloatingSummon.tsx – Breathing Summon Orb v2
// Deeper breathing, inner ripple, outer halo bloom on gesture, valence glow
// MIT License – Autonomicity Games Inc. 2026

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { currentValence } from '@/core/valence-tracker';
import mercyHaptic from '@/utils/haptic-utils';

const FloatingSummon: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [valence, setValence] = useState(currentValence.get());

  useEffect(() => {
    const unsubscribe = currentValence.subscribe(v => {
      setValence(v);
      if (v > 0.98) {
        setIsActive(true);
        mercyHaptic.playPattern('eternalHarmony', v);
      } else {
        setIsActive(false);
      }
    });
    return unsubscribe;
  }, []);

  const orbVariants = {
    idle: { scale: 1, opacity: 0.7 },
    breathe: {
      scale: [1, 1.12, 1],
      opacity: [0.7, 1, 0.7],
      transition: { duration: 5, repeat: Infinity, ease: "easeInOut" }
    },
    bloom: {
      scale: [1, 1.3, 1.1],
      opacity: [0.7, 1, 0.9],
      transition: { duration: 2, ease: "easeOut" }
    }
  };

  return (
    <motion.div
      className="fixed bottom-12 right-12 z-50 cursor-pointer"
      onClick={() => {
        setIsActive(!isActive);
        mercyHaptic.playPattern(isActive ? 'neutralPulse' : 'cosmicHarmony', valence);
      }}
      variants={orbVariants}
      animate={isActive ? "breathe" : "idle"}
      whileHover={{ scale: 1.15 }}
      whileTap={{ scale: 0.95 }}
    >
      <motion.div
        className="relative w-24 h-24 rounded-full bg-gradient-to-br from-cyan-500/30 to-emerald-500/30 backdrop-blur-2xl border border-cyan-400/40 shadow-2xl flex items-center justify-center"
        animate={{ rotate: 360 }}
        transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
      >
        <motion.div
          className="absolute inset-2 rounded-full border-2 border-emerald-400/60"
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
        <div className="text-2xl font-light text-white/90">
          Mercy
        </div>
      </motion.div>

      {isActive && (
        <motion.div
          className="absolute inset-[-40px] rounded-full bg-gradient-radial from-cyan-400/20 to-transparent pointer-events-none"
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1.5, opacity: 0.6 }}
          transition={{ duration: 1.5, repeat: Infinity, repeatType: "reverse" }}
        />
      )}
    </motion.div>
  );
};

export default FloatingSummon;
