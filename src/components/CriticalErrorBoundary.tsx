// src/components/CriticalErrorBoundary.tsx – Critical error boundary with mercy UI
// Catches render errors & shows graceful fallback
// MIT License – Autonomicity Games Inc. 2026

import React from 'react'

interface Props {
  children: React.ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

class CriticalErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[CriticalErrorBoundary] Render error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          height: '100vh',
          background: '#000',
          color: '#ff4444',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          textAlign: 'center',
          fontFamily: 'Courier New, monospace'
        }}>
          <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
            Lattice perturbation detected...
          </h1>
          <p style={{ fontSize: '1.2rem', maxWidth: '600px', opacity: 0.9 }}>
            Mercy gate activated. The lattice is self-healing.
            Please hard refresh (pull down or Ctrl+Shift+R).
          </p>
          {this.state.error && (
            <pre style={{
              background: '#111',
              padding: '1rem',
              borderRadius: '8px',
              marginTop: '2rem',
              maxWidth: '90%',
              overflow: 'auto'
            }}>
              {this.state.error.message}
            </pre>
          )}
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '2rem',
              padding: '1rem 2rem',
              background: '#00ff88',
              color: '#000',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1.2rem',
              cursor: 'pointer'
            }}
          >
            Re-enter Lattice
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

export default CriticalErrorBoundary
