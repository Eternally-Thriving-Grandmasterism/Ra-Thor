const StyleDictionary = require('style-dictionary');

module.exports = {
  source: ['design-tokens/tokens.json'],
  platforms: {
    css: {
      transformGroup: 'css',
      buildPath: 'design-tokens/css/',
      files: [{
        destination: 'variables.css',
        format: 'css/variables',
        options: {
          outputReferences: true
        }
      }]
    }
  }
};