// vite.config.ts – Mobile-First Vite Config v2.4
// Aggressive mobile optimization: chunks, preloads, compression, PWA tuning
// MIT License – Autonomicity Games Inc. 2026

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import viteCompression from 'vite-plugin-compression2'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg', 'pwa-*.png'],
      manifest: {
        name: 'Rathor — Mercy Strikes First',
        short_name: 'Rathor',
        description: 'Sovereign offline AGI lattice — eternal thriving through valence-locked truth',
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
    }),
    viteCompression({
      algorithm: 'brotliCompress',
      exclude: [/\.html$/], // HTML is already small
      threshold: 1024 * 10, // 10 KB
      compressionOptions: { level: 11 }
    }),
    viteCompression({
      algorithm: 'gzip',
      exclude: [/\.html$/],
      threshold: 1024 * 10
    })
  ],

  base: '/Rathor-NEXi/',

  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: false,
        passes: 3,
        pure_funcs: ['console.debug'],
        pure_getters: true,
        unsafe: true,
        unsafe_comps: true,
        unsafe_math: true,
        unsafe_methods: true,
        unsafe_undefined: true
      },
      mangle: true,
      format: { comments: false }
    },
    rollupOptions: {
      output: {
        // Mobile-first manual chunking
        manualChunks: {
          // Core vendor (loads first)
          vendor: ['react', 'react-dom', 'framer-motion'],

          // Heavy ML deps (lazy loaded)
          tfjs: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl'],
          mediapipe: ['@mediapipe/holistic'],

          // Other utils
          utils: ['./src/utils/haptic-utils.ts', './src/core/valence-tracker.ts']
        },
        // Better mobile caching
        entryFileNames: 'assets/entry/[name]-[hash].js',
        chunkFileNames: 'assets/chunks/[name]-[hash].js',
        assetFileNames: 'assets/static/[name]-[hash][extname]'
      }
    },
    target: 'es2020', // modern mobile browsers
    cssCodeSplit: true,
    reportCompressedSize: true
  },

  server: {
    port: 3000,
    open: true,
    hmr: true
  },

  preview: {
    port: 4173
  },

  // Mobile-first dev optimizations
  optimizeDeps: {
    include: [
      'react', 'react-dom', 'framer-motion',
      '@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl',
      '@mediapipe/holistic'
    ]
  },

  // Preload heavy deps in dev
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  }
})
          // MediaPipe Holistic (WASM heavy – already lazy)
          mediapipe: ['@mediapipe/holistic'],

          // Other heavy libs (if any added later)
          utils: ['./src/utils/haptic-utils.ts', './src/core/valence-tracker.ts']
        },

        // Better chunk naming & size control
        chunkFileNames: 'assets/chunks/[name]-[hash].js',
        entryFileNames: 'assets/entry/[name]-[hash].js',
        assetFileNames: 'assets/static/[name]-[hash][extname]'
      }
    },
    target: 'es2020',           // modern browsers – smaller polyfills
    cssCodeSplit: true,         // split CSS per chunk
    reportCompressedSize: true  // show brotli/gzip sizes in build output
  },

  server: {
    port: 3000,
    open: true,
    hmr: true
  },

  preview: {
    port: 4173
  },

  // Enable fast refresh & better dev HMR
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  },

  // Preload heavy deps in dev
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-webgl',
      '@mediapipe/holistic'
    ]
  }
})
