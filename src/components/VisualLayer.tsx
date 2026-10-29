// src/components/VisualLayer.tsx – Visuals & heavy components
// Orb, particle field, dashboard – loaded after core
// MIT License – Autonomicity Games Inc. 2026

import React from 'react'

const VisualLayer: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div style={{ height: '100%', position: 'relative' }}>
      {/* Floating summon orb, particle field, etc. can be added here */}
      {children}
    </div>
  )
}

export default VisualLayer
