// src/components/CriticalCore.tsx – Loads essential core modules first
// Valence tracker, mercy gate, haptic utils, etc.
// MIT License – Autonomicity Games Inc. 2026

import React, { Suspense } from 'react'
import { currentValence } from '@/core/valence-tracker'
import { mercyGate } from '@/core/mercy-gate'

const CriticalCore: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Force early initialization of critical core
  useEffect(() => {
    // Pre-warm valence & mercy systems
    currentValence.get()
    mercyGate('CriticalCore init').catch(console.error)
  }, [])

  return (
    <Suspense fallback={<div style={{ color: '#ff8800' }}>Core systems initializing...</div>}>
      {children}
    </Suspense>
  )
}

export default CriticalCore
