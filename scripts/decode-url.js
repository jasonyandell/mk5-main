#!/usr/bin/env node

// Decode game state from URL parameter
const input = process.argv[2];
if (!input) {
  console.error('Usage: node decode-url.js <base64-encoded-state>');
  process.exit(1);
}

try {
  const decoded = Buffer.from(input, 'base64').toString('utf-8');
  const parsed = JSON.parse(decoded);
  
  console.log('=== GAME STATE ===');
  console.log('Version:', parsed.v);
  console.log('Seed:', parsed.s?.s);
  console.log('\n=== ACTIONS ===');
  
  parsed.a?.forEach((action, index) => {
    console.log(`${index + 1}. ${action.i}`);
  });
  
  console.log('\n=== RAW JSON ===');
  console.log(JSON.stringify(parsed, null, 2));
} catch (e) {
  console.error('Failed to decode:', e.message);
  process.exit(1);
}