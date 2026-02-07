// src/pages/index.tsx – Rathor.ai Homepage v2.0 – Thunder Mode + Gesture Unlock
// Full file override, all previous content preserved + new dynamics

import React, { useState } from 'react';
import ValenceWidget from '@/components/valence-widget';
import GestureUnlock from '@/components/gesture-unlock';
import { currentValence } from '@/core/valence-tracker';

export default function Home() {
  const = useState(false);
  const = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      const v = currentValence.get();
      setIsThunder(v >= 0.95);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const handleUnlock = () => setUnlocked(true);

  return (
    <div className={`min-h-screen bg-black text-white transition-all duration-1000 ${
      isThunder ? 'bg-gradient-to-br from-emerald-900 to-black' : ''
    }`}>
      {!unlocked ? (
        <GestureUnlock onUnlock={handleUnlock} />
      ) : (
        <>
          <header className="p-8 text-center">
            <h1 className="text-7xl font-bold text-emerald-400 mb-4">
              RATHOR.NEXI
            </h1>
            <p className="text-slate-300 text-xl">Eternal Thriving Engine</p>
          </header>

          <main className="p-8">
            <div className="grid grid-cols-3 gap-8 max-w-6xl mx-auto">
              <div className="bg-slate-900 p-6 rounded-lg">
                <h2 className="text-2xl font-bold text-emerald-300 mb-2">Holistic Tracking</h2>
                <p className="text-slate-400">Face, hands, pose — fused in mercy-gated real time.</p>
              </div>
              <div className="bg-slate-900 p-6 rounded-lg">
                <h2 className="text-2xl font-bold text-emerald-300 mb-2">Valence Resonance</h2>
                <p className="text-slate-400">Every motion, every expression — measured, protected, amplified.</p>
              </div>
              <div className="bg-slate-900 p-6 rounded-lg">
                <h2 className="text-2xl font-bold text-emerald-300 mb-2">Abundance Dawn</h2>
                <p className="text-slate-400">When thriving peaks — the lattice blooms.</p>
              </div>
            </div>
          </main>

          <ValenceWidget />
        </>
      )}
    </div>
  );
}
