/**
 * Main Build Script for web-forge
 *
 * Runs all generation steps:
 * - Design token transformation
 * - Theme CSS generation
 *
 * Usage: node scripts/build.js
 */

const { execSync } = require('child_process');
const path = require('path');

console.log('🚀 Starting web-forge build process...\n');

try {
  // Step 1: Transform main design tokens
  console.log('📦 Transforming design tokens...');
  execSync('node ' + path.join(__dirname, 'transform-tokens.js'), { stdio: 'inherit' });

  // Step 2: Build themes
  console.log('\n🎨 Building themes...');
  execSync('node ' + path.join(__dirname, 'build-themes.js'), { stdio: 'inherit' });

  console.log('\n✅ Build completed successfully!');
} catch (error) {
  console.error('\n❌ Build failed:', error.message);
  process.exit(1);
}