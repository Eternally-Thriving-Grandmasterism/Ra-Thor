import { defineConfig, splitVendorChunkPlugin } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import viteCompression from 'vite-plugin-compression2'
import path from 'path'

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    splitVendorChunkPlugin(),
    VitePWA({
      registerType: 'autoUpdate',
      devOptions: { enabled: true },
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'pwa-*.png', '**/*.wasm'],
      manifest: {
        name: 'Rathor — Mercy Strikes First',
        short_name: 'Rathor',
        description: 'Sovereign offline AGI lattice',
        theme_color: '#00ff88',
        background_color: '#000000',
        display: 'standalone',
        scope: '/',
        start_url: '/',
        icons: [
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff,woff2,wasm,onnx}'],
        // Runtime caching strategies
        runtimeCaching: [
          // Cache Mediapipe WASM & JS – CacheFirst, long-lived
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@mediapipe\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'mediapipe-runtime',
              expiration: { maxEntries: 30, maxAgeSeconds: 60 * 60 * 24 * 30 }, // 30 days
              cacheableResponse: { statuses: [0, 200] }
            }
          },
          // tfjs assets – CacheFirst
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@tensorflow\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'tfjs-runtime',
              expiration: { maxEntries: 60, maxAgeSeconds: 60 * 60 * 24 * 30 }
            }
          },
          // Our own static assets – StaleWhileRevalidate for fast repeat visits
          {
            urlPattern: /^https:\/\/eternally-thriving-grandmasterism\.github\.io\/Rathor-NEXi\/.*/i,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'rathor-static',
              expiration: { maxEntries: 100, maxAgeSeconds: 60 * 60 * 24 * 7 } // 1 week
            }
          }
        ],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api\//, /\.wasm$/],
        cleanupOutdatedCaches: true,
        skipWaiting: true,
        clientsClaim: true
      }
    }),
    viteCompression({ algorithm: 'brotliCompress', exclude: [/\.html$/], threshold: 10240 }),
    viteCompression({ algorithm: 'gzip', exclude: [/\.html$/], threshold: 10240 })
  ],

  base: '/Rathor-NEXi/',

  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') }
  },

  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: mode === 'development',
    minify: 'terser',
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-core': ['react', 'react-dom'],
          'vendor-ml': ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl'],
          'vendor-mediapipe': ['@mediapipe/holistic']
        }
      }
    }
  },

  server: { port: 3000, open: true, hmr: true, fs: { strict: false } },
  preview: { port: 4173 },

  optimizeDeps: {
    include: ['react', 'react-dom', '@tensorflow/tfjs', '@mediapipe/holistic'],
    exclude: ['onnxruntime-web']
  },

  logLevel: 'info'
}))
