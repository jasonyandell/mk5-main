#!/usr/bin/env node

// Get game state from URL parameter using the replay utility
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const input = process.argv[2];
if (!input) {
  console.error('Usage: node scripts/get-state-from-url.js <url-or-base64>');
  console.error('Example: node scripts/get-state-from-url.js http://localhost:3000/?d=eyJ2IjoxLCJzIjp7InMiOjEyMzQ1fSwiYSI6W3siaSI6IjMwIn1dfQ==');
  console.error('     or: node scripts/get-state-from-url.js eyJ2IjoxLCJzIjp7InMiOjEyMzQ1fSwiYSI6W3siaSI6IjMwIn1dfQ==');
  process.exit(1);
}

try {
  // Create a temporary TypeScript file to execute using the replay utility
  const tempScript = `
import { replayFromUrl } from '../src/game/utils/replay';

const url = ${JSON.stringify(input)};

const result = replayFromUrl(url);

if (result.errors && result.errors.length > 0) {
  console.error('Replay errors:', result.errors);
  process.exit(1);
}

console.log(JSON.stringify(result.state, null, 2));
`;

  // Write to temp file and execute
  const tempFile = join(__dirname, '..', 'scratch', 'temp-get-state.ts');
  
  // Ensure scratch directory exists
  const scratchDir = join(__dirname, '..', 'scratch');
  if (!fs.existsSync(scratchDir)) {
    fs.mkdirSync(scratchDir, { recursive: true });
  }
  
  fs.writeFileSync(tempFile, tempScript);
  
  // Execute with tsx
  const result = execSync(`npx tsx ${tempFile}`, {
    encoding: 'utf8',
    cwd: join(__dirname, '..'),
    stdio: ['pipe', 'pipe', 'pipe']
  });
  
  // Clean up temp file
  fs.unlinkSync(tempFile);
  
  console.log(result);
} catch (e) {
  console.error('Error:', e.message);
  process.exit(1);
}