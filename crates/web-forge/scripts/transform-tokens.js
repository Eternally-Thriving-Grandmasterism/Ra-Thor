/**
 * Simple Design Token Transformation Script
 * For web-forge
 *
 * Usage: node scripts/transform-tokens.js
 */

const fs = require('fs');
const path = require('path');

const tokensPath = path.join(__dirname, '../design-tokens/tokens.json');
const outputPath = path.join(__dirname, '../design-tokens/css/variables.css');

const tokens = JSON.parse(fs.readFileSync(tokensPath, 'utf8')).RaThor;

let css = ':root {\n';

// Simple flattener for semantic tokens
Object.keys(tokens.color.semantic).forEach(category => {
  Object.keys(tokens.color.semantic[category]).forEach(key => {
    const varName = `--color-${category}-${key}`;
    const value = tokens.color.semantic[category][key].value;
    css += `  ${varName}: ${value};\n`;
  });
});

css += '}\n';

fs.writeFileSync(outputPath, css);
console.log('✅ Design tokens transformed to CSS variables.');