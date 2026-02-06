import React, { Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

const LoadingFallback = () => (
  <div style={{
    height: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#000',
    color: '#00ff88',
    fontSize: '1.8rem',
    textShadow: '0 0 10px #00ff88',
    flexDirection: 'column',
    gap: '1rem'
  }}>
    <div>Thunder eternal surges through the lattice...</div>
    <div style={{ width: '300px', height: '6px', background: '#111', borderRadius: '3px', overflow: 'hidden' }}>
      <div style={{
        height: '100%',
        width: '0%',
        background: 'linear-gradient(90deg, #00ff88, #00aaff)',
        animation: 'progress 4s linear infinite'
      }} />
    </div>
  </div>
)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Suspense fallback={<LoadingFallback />}>
      <App />
    </Suspense>
  </React.StrictMode>
)
