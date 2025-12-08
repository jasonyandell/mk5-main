#!/usr/bin/env npx tsx
/**
 * MCCFR Training Script
 *
 * Trains a Monte Carlo CFR strategy for Texas 42 trick-taking.
 *
 * Usage:
 *   npx tsx scripts/train-mccfr.ts [options]
 *
 * Options:
 *   --iterations <n>    Number of training iterations (default: 10000)
 *   --seed <n>          Random seed (default: 42)
 *   --output <file>     Output file (default: trained-strategy.json)
 *   --progress          Show progress updates
 *   --quiet             Suppress output except errors
 */

import { MCCFRTrainer } from '../src/game/ai/cfr';
import * as fs from 'fs';
import * as path from 'path';

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    iterations: 10000,
    seed: 42,
    output: 'trained-strategy.json',
    progress: false,
    quiet: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case '--iterations':
      case '-i':
        config.iterations = parseInt(next ?? '10000', 10);
        i++;
        break;
      case '--seed':
      case '-s':
        config.seed = parseInt(next ?? '42', 10);
        i++;
        break;
      case '--output':
      case '-o':
        config.output = next ?? 'trained-strategy.json';
        i++;
        break;
      case '--progress':
      case '-p':
        config.progress = true;
        break;
      case '--quiet':
      case '-q':
        config.quiet = true;
        break;
      case '--help':
      case '-h':
        console.log(`
MCCFR Training Script

Usage:
  npx tsx scripts/train-mccfr.ts [options]

Options:
  --iterations, -i <n>    Number of training iterations (default: 10000)
  --seed, -s <n>          Random seed (default: 42)
  --output, -o <file>     Output file (default: trained-strategy.json)
  --progress, -p          Show progress updates
  --quiet, -q             Suppress output except errors
  --help, -h              Show this help message

Examples:
  npx tsx scripts/train-mccfr.ts --iterations 100000 --progress
  npx tsx scripts/train-mccfr.ts -i 50000 -o my-strategy.json
        `);
        process.exit(0);
    }
  }

  return config;
}

async function main() {
  const config = parseArgs();

  if (!config.quiet) {
    console.log('MCCFR Training for Texas 42');
    console.log('============================');
    console.log(`Iterations: ${config.iterations.toLocaleString()}`);
    console.log(`Seed: ${config.seed}`);
    console.log(`Output: ${config.output}`);
    console.log('');
  }

  const startTime = Date.now();

  const trainerConfig: Partial<import('../src/game/ai/cfr').MCCFRConfig> = {
    iterations: config.iterations,
    seed: config.seed
  };

  if (config.progress) {
    trainerConfig.progressInterval = 10; // Report every 10 iterations
    if (!config.quiet) {
      trainerConfig.onProgress = (iter, total, nodes) => {
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = Math.round(iter / elapsed);
        const eta = Math.round((total - iter) / rate);
        process.stdout.write(
          `\rProgress: ${iter.toLocaleString()}/${total.toLocaleString()} ` +
          `(${Math.round(100 * iter / total)}%) | ` +
          `${nodes.toLocaleString()} info sets | ` +
          `${rate.toLocaleString()} iter/s | ` +
          `ETA: ${eta}s   `
        );
      };
    }
  }

  const trainer = new MCCFRTrainer(trainerConfig);

  if (!config.quiet) {
    console.log('Training started...');
  }

  const result = await trainer.train();

  if (config.progress && !config.quiet) {
    process.stdout.write('\n');
  }

  if (!config.quiet) {
    console.log('');
    console.log('Training complete!');
    console.log('------------------');
    console.log(`Iterations: ${result.iterations.toLocaleString()}`);
    console.log(`Info sets: ${result.infoSetCount.toLocaleString()}`);
    console.log(`Time: ${(result.trainingTimeMs / 1000).toFixed(1)}s`);
    console.log(`Rate: ${result.iterationsPerSecond.toLocaleString()} iter/s`);
    console.log('');
  }

  // Serialize and save
  const serialized = trainer.serialize();
  const outputPath = path.resolve(config.output);

  fs.writeFileSync(outputPath, JSON.stringify(serialized, null, 2));

  if (!config.quiet) {
    const stats = fs.statSync(outputPath);
    console.log(`Saved to: ${outputPath}`);
    console.log(`File size: ${(stats.size / 1024).toFixed(1)} KB`);
  }

  // Print regret table stats
  const tableStats = trainer.getRegretTable().getStats();
  if (!config.quiet) {
    console.log('');
    console.log('Regret Table Stats:');
    console.log(`  Nodes: ${tableStats.nodeCount.toLocaleString()}`);
    console.log(`  Total visits: ${tableStats.totalVisits.toLocaleString()}`);
    console.log(`  Avg actions/node: ${tableStats.avgActionsPerNode.toFixed(2)}`);
  }
}

main().catch(err => {
  console.error('Training failed:', err);
  process.exit(1);
});
