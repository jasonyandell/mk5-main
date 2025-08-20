#!/usr/bin/env node

// Use dynamic import with tsx to handle TypeScript
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Parse command line arguments
const args = process.argv.slice(2);
const url = args[0];

if (!url) {
  console.error('Usage: node scripts/replay-from-url.js <url-or-base64> [options]');
  console.error('\nOptions:');
  console.error('  --stop-at N         Stop replay at action N');
  console.error('  --verbose           Show each action as it\'s replayed');
  console.error('  --generate-test     Generate a test file in scratch/');
  console.error('  --action-range S E  Show details only for actions S to E');
  console.error('  --show-tricks       Display trick-by-trick breakdown');
  console.error('  --hand N            Focus on just hand N');
  console.error('  --compact           One-line-per-action format');
  console.error('\nExamples:');
  console.error('  node scripts/replay-from-url.js http://localhost:3000/?d=...');
  console.error('  node scripts/replay-from-url.js <url> --show-tricks');
  console.error('  node scripts/replay-from-url.js <url> --action-range 87 92');
  console.error('  node scripts/replay-from-url.js <url> --hand 4');
  console.error('  node scripts/replay-from-url.js <url> --compact');
  process.exit(1);
}

// Parse options from command line
let stopAt = undefined;
let verbose = false;
let generateTest = false;
let actionRangeStart = undefined;
let actionRangeEnd = undefined;
let showTricks = false;
let focusHand = undefined;
let compact = false;

for (let i = 1; i < args.length; i++) {
  if (args[i] === '--stop-at' && args[i + 1]) {
    stopAt = parseInt(args[i + 1], 10);
    i++;
  } else if (args[i] === '--verbose') {
    verbose = true;
  } else if (args[i] === '--generate-test') {
    generateTest = true;
  } else if (args[i] === '--action-range' && args[i + 1] && args[i + 2]) {
    actionRangeStart = parseInt(args[i + 1], 10);
    actionRangeEnd = parseInt(args[i + 2], 10);
    i += 2;
  } else if (args[i] === '--show-tricks') {
    showTricks = true;
  } else if (args[i] === '--hand' && args[i + 1]) {
    focusHand = parseInt(args[i + 1], 10);
    i++;
  } else if (args[i] === '--compact') {
    compact = true;
  }
}

// Create TypeScript code to execute
const tempScript = `
import { replayFromUrl } from '../src/game/utils/replay';

const url = ${JSON.stringify(url)};
const options = {
  stopAt: ${stopAt !== undefined ? stopAt : 'undefined'},
  verbose: ${verbose},
  actionRangeStart: ${actionRangeStart !== undefined ? actionRangeStart : 'undefined'},
  actionRangeEnd: ${actionRangeEnd !== undefined ? actionRangeEnd : 'undefined'},
  showTricks: ${showTricks},
  focusHand: ${focusHand !== undefined ? focusHand : 'undefined'},
  compact: ${compact}
};

try {
  console.log('Replaying game from URL...\\n');
  
  const result = replayFromUrl(url, options);
  
  if (result.errors && result.errors.length > 0) {
    console.error('\\n❌ Replay encountered errors:');
    result.errors.forEach(err => console.error(\`  \${err}\`));
  }
  
  console.log(\`\\n✅ Replayed \${result.actionCount} actions successfully\`);
  console.log('\\n=== Final Game State ===');
  console.log('Phase:', result.state.phase);
  console.log('Team Scores:', result.state.teamScores);
  console.log('Team Marks:', result.state.teamMarks);
  console.log('Hands Played:', result.state.teamMarks[0] + result.state.teamMarks[1]);
  
  if (result.state.phase === 'playing' || result.state.phase === 'scoring') {
    console.log('\\n=== Current Hand ===');
    console.log('Dealer: Player', result.state.dealer);
    console.log('Winning Bidder: Player', result.state.winningBidder);
    console.log('Bid:', result.state.currentBid?.value || 'N/A');
    console.log('Trump:', result.state.trump.type === 'suit' ? \`Suit \${result.state.trump.suit}\` : result.state.trump.type);
    console.log('Tricks Played:', result.state.tricks.length);
    
    if (result.state.tricks.length > 0) {
      let team0Points = 0;
      let team1Points = 0;
      
      result.state.tricks.forEach((trick, i) => {
        const winnerTeam = trick.winner % 2 === 0 ? 0 : 1;
        if (winnerTeam === 0) {
          team0Points += trick.points;
        } else {
          team1Points += trick.points;
        }
      });
      
      console.log('Points Won So Far: Team 0:', team0Points, ', Team 1:', team1Points);
    }
  }
  
  if (${verbose} || ${stopAt !== undefined}) {
    console.log('\\n=== Full State (JSON) ===');
    console.log(JSON.stringify(result.state, null, 2));
  }
  
} catch (error) {
  console.error('❌ Error:', error.message);
  if (${verbose}) {
    console.error(error.stack);
  }
  process.exit(1);
}
`;

// Ensure scratch directory exists
const scratchDir = join(__dirname, '..', 'scratch');
if (!fs.existsSync(scratchDir)) {
  fs.mkdirSync(scratchDir, { recursive: true });
}

// If generate-test flag is set, create a test file instead
if (generateTest) {
  // Extract and decode the URL data
  let base64Data;
  if (url.includes('://') || url.startsWith('localhost')) {
    const urlObj = new URL(url.startsWith('localhost') ? 'http://' + url : url);
    base64Data = urlObj.searchParams.get('d');
  } else {
    base64Data = url;
  }
  
  if (!base64Data) {
    console.error('No "d" parameter found in URL');
    process.exit(1);
  }
  
  const decoded = Buffer.from(base64Data, 'base64').toString('utf-8');
  const urlData = JSON.parse(decoded);
  
  // Generate test file content
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const testFile = join(scratchDir, `test-${timestamp}.js`);
  
  const testContent = `// Test generated from URL replay
// Run with: npx tsx scratch/test-{timestamp}.js
// Or convert to vitest test by wrapping in describe/it blocks

import { createInitialState, getNextStates, decompressActionId } from '../src/game/index.js';

function runTest() {
    const seed = ${urlData.s?.s};
    const compressedActions = ${JSON.stringify(urlData.a?.map(a => a.i) || [], null, 2).split('\n').map((line, i) => i === 0 ? line : '      ' + line).join('\n')};
    
    // Replay the game
    let state = createInitialState({ shuffleSeed: seed });
    const errors = [];
    
    for (let i = 0; i < compressedActions.length; i++) {
      const actionId = compressedActions[i];
      const decompressedId = decompressActionId(actionId);
      const availableTransitions = getNextStates(state);
      const matchingTransition = availableTransitions.find(t => t.id === decompressedId);
      
      if (!matchingTransition) {
        errors.push(\`Action \${i}: Invalid action "\${decompressedId}" at phase \${state.phase}\`);
        break;
      }
      
      state = matchingTransition.newState;
    }
    
    // Final state snapshot
    console.log('\\n=== Final State ===');
    console.log('Phase:', state.phase);
    console.log('Team Scores:', state.teamScores);
    console.log('Team Marks:', state.teamMarks);
    console.log('Hands Played:', state.handsPlayed);
    
    if (state.phase === 'playing' || state.phase === 'scoring') {
      console.log('\\n=== Current Hand ===');
      console.log('Dealer:', state.dealer);
      console.log('Bidder:', state.winningBidder);
      console.log('Bid:', state.currentBid?.value);
      console.log('Trump:', state.trump);
    }
    
    // ADD YOUR ASSERTIONS HERE
    // Example: Check if the score is what you expect
    // if (state.teamScores[0] !== 17 || state.teamScores[1] !== 22) {
    //   throw new Error(\`Expected scores [17, 22] but got [\${state.teamScores}]\`);
    // }
    
    // To replay to a specific action:
    // let midState = createInitialState({ shuffleSeed: seed });
    // for (let i = 0; i < 50; i++) { // stop at action 50
    //   const actionId = compressedActions[i];
    //   const decompressedId = decompressActionId(actionId);
    //   const availableTransitions = getNextStates(midState);
    //   const matchingTransition = availableTransitions.find(t => t.id === decompressedId);
    //   if (matchingTransition) midState = matchingTransition.newState;
    // }
    
    return { state, errors };
}

// Run the test
try {
  const result = runTest();
  if (result.errors.length > 0) {
    console.error('\\n❌ Test failed with errors:', result.errors);
    process.exit(1);
  }
  console.log('\\n✅ Test completed successfully');
} catch (error) {
  console.error('\\n❌ Test failed:', error.message);
  process.exit(1);
}
`;
  
  fs.writeFileSync(testFile, testContent);
  console.log(`✅ Generated test file: ${testFile}`);
  console.log('\nTo run the test:');
  console.log(`  npx tsx ${testFile}`);
  console.log('\nNext steps:');
  console.log('1. Add assertions for the specific bug you\'re testing');
  console.log('2. Run the test to verify it catches the issue');
  console.log('3. Fix the bug in the game logic');
  console.log('4. Re-run the test to confirm the fix');
  process.exit(0);
}

// Write to temp file and execute
const tempFile = join(__dirname, '..', 'scratch', 'temp-replay.ts');

fs.writeFileSync(tempFile, tempScript);

try {
  // Execute with tsx
  const result = execSync(`npx tsx ${tempFile}`, {
    encoding: 'utf8',
    cwd: join(__dirname, '..'),
    stdio: 'inherit'
  });
} finally {
  // Clean up temp file
  fs.unlinkSync(tempFile);
}