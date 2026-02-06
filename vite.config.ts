import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg', 'pwa-*.png'],
      manifest: {
        name: 'Rathor — Mercy Strikes First',
        short_name: 'Rathor',
        description: 'Mercy-gated symbolic AGI lattice — eternal thriving through valence-locked truth',
        theme_color: '#00ff88',
        background_color: '#000000',
        display: 'standalone',
        scope: '/Rathor-NEXi/',
        start_url: '/Rathor-NEXi/',
        icons: [
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
            handler: 'CacheFirst',
            options: { cacheName: 'cdn-assets', expiration: { maxEntries: 50, maxAgeSeconds: 2592000 } }
          },
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|ico)$/,
            handler: 'CacheFirst',
            options: { cacheName: 'images', expiration: { maxEntries: 100, maxAgeSeconds: 2592000 } }
          }
        ],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api\//]
      },
      devOptions: { enabled: true }
    })
  ],
  base: '/Rathor-NEXi/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'terser',
    terserOptions: {
      compress: { drop_console: false, passes: 3 },
      mangle: true
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'framer-motion'],
          tfjs: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl'],
          mediapipe: ['@mediapipe/holistic']
        }
      }
    }
  },
  server: { port: 3000, open: true },
  preview: { port: 4173 }
})
