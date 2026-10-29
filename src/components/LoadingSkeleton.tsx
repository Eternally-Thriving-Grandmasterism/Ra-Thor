// src/components/LoadingSkeleton.tsx – Optimized loading skeleton v1
// Progressive reveal + shimmer + valence-aware pulse
// MIT License – Autonomicity Games Inc. 2026

import React from 'react'

const LoadingSkeleton: React.FC = () => {
  return (
    <div style={{
      height: '100vh',
      background: '#000',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#00ff88',
      fontFamily: 'Courier New, monospace',
      gap: '2rem'
    }}>
      <div style={{
        fontSize: '2.5rem',
        textShadow: '0 0 15px #00ff88',
        animation: 'pulse 2s infinite'
      }}>
        Lattice awakening...
      </div>

      <div style={{
        width: '320px',
        height: '8px',
        background: '#111',
        borderRadius: '4px',
        overflow: 'hidden',
        position: 'relative'
      }}>
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(90deg, #00ff88, #00aaff, #00ff88)',
          animation: 'shimmer 2.5s infinite linear'
        }} />
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '1.5rem',
        width: '90%',
        maxWidth: '400px'
      }}>
        {Array(4).fill(0).map((_, i) => (
          <div
            key={i}
            style={{
              height: '120px',
              background: '#111',
              borderRadius: '12px',
              animation: 'shimmer 2s infinite linear',
              animationDelay: `${i * 0.15}s`
            }}
          />
        ))}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.7; }
          50% { opacity: 1; }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}

export default LoadingSkeleton
