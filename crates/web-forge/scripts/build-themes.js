/**
 * Theme Build Script
 * Generates theme-specific CSS files from central theme definitions.
 *
 * Usage: node scripts/build-themes.js
 */

const fs = require('fs');
const path = require('path');

const themesDir = path.join(__dirname, '../design-tokens/themes');
const outputDir = path.join(__dirname, '../design-tokens/css/themes');

// Ensure output directory exists
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

function generateThemeCSS(themeName, themeData) {
  let css = `:root[data-theme="${themeName}"] {\n`;

  if (themeData.colors) {
    Object.keys(themeData.colors).forEach(category => {
      Object.keys(themeData.colors[category]).forEach(key => {
        const varName = `--color-${category}-${key}`;
        const value = themeData.colors[category][key];
        css += `  ${varName}: ${value};\n`;
      });
    });
  }

  css += `}\n`;
  return css;
}

// Read all theme JSON files
fs.readdirSync(themesDir).forEach(file => {
  if (file.endsWith('.json')) {
    const themeName = path.basename(file, '.json');
    const themePath = path.join(themesDir, file);
    const themeData = JSON.parse(fs.readFileSync(themePath, 'utf8'));

    const cssContent = generateThemeCSS(themeName, themeData);
    const outputPath = path.join(outputDir, `${themeName}.css`);

    fs.writeFileSync(outputPath, cssContent);
    console.log(`✅ Generated theme: ${themeName}.css`);
  }
});

console.log('\nTheme generation complete.');