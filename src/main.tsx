// src/main.tsx – Entry point with ultra-granular Suspense boundaries v4
// Progressive layered fallbacks + granular lazy loading + debug timing
// MIT License – Autonomicity Games Inc. 2026

import React, { Suspense, lazy, useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

// ─── Ultra-granular lazy components ─────────────────────────────────
const CriticalCore = lazy(() => import('./CriticalCore.tsx'))           // valence-tracker, mercy-gate, core utils
const VisualLayer = lazy(() => import('./VisualLayer.tsx'))           // orb, particles, dashboard shell
const App = lazy(() => import('./App.tsx'))                           // main app logic + heavy ML/gesture

// ─── Layered fallback components ────────────────────────────────────
const UltraLightFallback = () => (
  <div style={{
    height: '100vh',
    background: '#000',
    color: '#00ff88',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.8rem',
    textShadow: '0 0 10px #00ff88',
    gap: '1.5rem',
    textAlign: 'center',
    padding: '1rem'
  }}>
    <div>Thunder eternal surges through the lattice...</div>
    <div style={{
      width: '300px',
      height: '6px',
      background: '#111',
      borderRadius: '3px',
      overflow: 'hidden'
    }}>
      <div style={{
        height: '100%',
        width: '0%',
        background: 'linear-gradient(90deg, #00ff88, #00aaff)',
        animation: 'progress 12s linear infinite'
      }} />
    </div>
    <style>{`
      @keyframes progress {
        0% { width: 0%; }
        100% { width: 100%; }
      }
    `}</style>
    <p style={{ fontSize: '1rem', opacity: 0.7 }}>
      Phase 1 – Core awakening...
    </p>
  </div>
)

const CoreLoadingFallback = () => (
  <div style={{
    height: '100vh',
    background: '#000',
    color: '#00ff88',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.6rem',
    gap: '2rem'
  }}>
    <div>Lattice core initializing...</div>
    <div style={{
      width: '320px',
      height: '8px',
      background: '#111',
      borderRadius: '4px',
      overflow: 'hidden'
    }}>
      <div style={{
        height: '100%',
        width: '0%',
        background: 'linear-gradient(90deg, #00ff88, #00aaff)',
        animation: 'progress 6s linear infinite'
      }} />
    </div>
    <p style={{ fontSize: '1rem', opacity: 0.7 }}>
      Phase 2 – Valence & mercy systems online...
    </p>
  </div>
)

const VisualLoadingFallback = () => (
  <div style={{
    height: '100vh',
    background: '#000',
    color: '#00ff88',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.4rem',
    gap: '1.5rem'
  }}>
    <div>Visual lattice blooming...</div>
    <div style={{
      width: '300px',
      height: '6px',
      background: '#111',
      borderRadius: '3px',
      overflow: 'hidden'
    }}>
      <div style={{
        height: '100%',
        width: '0%',
        background: 'linear-gradient(90deg, #00ff88, #00aaff)',
        animation: 'progress 4s linear infinite'
      }} />
    </div>
    <p style={{ fontSize: '1rem', opacity: 0.7 }}>
      Phase 3 – Summon orb & dashboard emerging...
    </p>
  </div>
)

const CriticalErrorFallback = ({ error }: { error: Error }) => (
  <div style={{
    height: '100vh',
    background: '#000',
    color: '#ff4444',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem',
    textAlign: 'center'
  }}>
    <h1 style={{ fontSize: '2.2rem', marginBottom: '1rem' }}>
      Lattice perturbation detected
    </h1>
    <p style={{ fontSize: '1.1rem', maxWidth: '600px', opacity: 0.9 }}>
      Mercy gate activated. Self-healing in progress.
    </p>
    <pre style={{
      background: '#111',
      padding: '1rem',
      borderRadius: '8px',
      marginTop: '1.5rem',
      maxWidth: '90%',
      overflow: 'auto',
      fontSize: '0.9rem'
    }}>
      {error.message}
    </pre>
    <button
      onClick={() => window.location.reload()}
      style={{
        marginTop: '2rem',
        padding: '0.8rem 1.8rem',
        background: '#00ff88',
        color: '#000',
        border: 'none',
        borderRadius: '8px',
        fontSize: '1.1rem',
        cursor: 'pointer'
      }}
    >
      Re-enter Lattice
    </button>
  </div>
)

// ─── Critical error boundary component ─────────────────────────────
class CriticalErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error?: Error }> {
  state = { hasError: false, error: undefined }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[CriticalErrorBoundary]', error, info)
  }

  render() {
    if (this.state.hasError) {
      return <CriticalErrorFallback error={this.state.error!} />
    }
    return this.props.children
  }
}

// ─── Root with layered Suspense ────────────────────────────────────
const Root = () => (
  <Suspense fallback={<UltraLightFallback />}>
    <CriticalErrorBoundary>
      <Suspense fallback={<CoreLoadingFallback />}>
        <CriticalCore>
          <Suspense fallback={<VisualLoadingFallback />}>
            <VisualLayer>
              <App />
            </VisualLayer>
          </Suspense>
        </CriticalCore>
      </Suspense>
    </CriticalErrorBoundary>
  </Suspense>
)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
)
