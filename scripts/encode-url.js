#!/usr/bin/env node

// Encode game state to URL parameter
// Usage: node encode-url.js <seed> [action1] [action2] ...
// Example: node encode-url.js 12345 30 p p p trump-blanks 32 63 33 31 complete-trick

const args = process.argv.slice(2);
if (!args.length) {
  console.error('Usage: node encode-url.js <seed> [action1] [action2] ...');
  console.error('Example: node encode-url.js 12345 30 p p p trump-blanks 32 63 33 31 complete-trick');
  process.exit(1);
}

const seed = parseInt(args[0]);
if (isNaN(seed)) {
  console.error('First argument must be a numeric seed');
  process.exit(1);
}

const actions = args.slice(1).map(a => ({ i: a }));

const urlData = {
  v: 1,
  s: { s: seed },
  a: actions
};

const base64 = Buffer.from(JSON.stringify(urlData)).toString('base64');
console.log(`http://localhost:60101/?d=${base64}`);