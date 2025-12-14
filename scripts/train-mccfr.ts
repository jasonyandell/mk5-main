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
 *   --iterations <n>         Number of training iterations (default: 10000)
 *   --seed <n>               Random seed (default: 42)
 *   --output <file>          Output file (default: trained-strategy.json.gz)
 *   --resume <file>          Resume training from checkpoint file
 *   --checkpoint-interval <n> Save checkpoint every N iterations (default: 1000)
 *   --no-checkpoint          Disable checkpoints
 *   --progress               Show progress updates
 *   --quiet                  Suppress output except errors
 */

import { MCCFRTrainer } from '../src/game/ai/cfr';
import type { SerializedStrategy } from '../src/game/ai/cfr';
import * as fs from 'fs';
import * as path from 'path';
import { createGzip, createGunzip } from 'zlib';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    iterations: 10000,
    seed: 42,
    output: 'trained-strategy.json.gz',
    resume: null as string | null,
    checkpointInterval: 1000,
    noCheckpoint: false,
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
        config.output = next ?? 'trained-strategy.json.gz';
        i++;
        break;
      case '--resume':
      case '-r':
        config.resume = next ?? null;
        i++;
        break;
      case '--checkpoint-interval':
      case '-c':
        config.checkpointInterval = parseInt(next ?? '1000', 10);
        i++;
        break;
      case '--no-checkpoint':
        config.noCheckpoint = true;
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
  --iterations, -i <n>         Number of training iterations (default: 10000)
  --seed, -s <n>               Random seed (default: 42)
  --output, -o <file>          Output file (default: trained-strategy.json.gz)
  --resume, -r <file>          Resume training from checkpoint file
  --checkpoint-interval, -c <n> Save checkpoint every N iterations (default: 1000)
  --no-checkpoint              Disable checkpoints
  --progress, -p               Show progress updates
  --quiet, -q                  Suppress output except errors
  --help, -h                   Show this help message

Examples:
  npx tsx scripts/train-mccfr.ts --iterations 100000 --progress
  npx tsx scripts/train-mccfr.ts -i 50000 -o my-strategy.json.gz
  npx tsx scripts/train-mccfr.ts --resume checkpoint.json.gz -i 100000

Partitioned training (run on different machines, then merge):
  Machine 1: npx tsx scripts/train-mccfr.ts --seed 0 -i 500000 -o part1.json.gz
  Machine 2: npx tsx scripts/train-mccfr.ts --seed 500000 -i 500000 -o part2.json.gz
  Combine:   npx tsx scripts/merge-strategies.ts --inputs part1.json.gz part2.json.gz -o combined.json.gz
        `);
        process.exit(0);
    }
  }

  return config;
}

/**
 * Load a strategy file (supports both .json and .json.gz)
 */
async function loadStrategy(filePath: string): Promise<SerializedStrategy> {
  const resolvedPath = path.resolve(filePath);

  if (filePath.endsWith('.gz')) {
    // Decompress gzipped file
    const compressed = fs.readFileSync(resolvedPath);
    const gunzip = createGunzip();
    const chunks: Buffer[] = [];

    return new Promise((resolve, reject) => {
      gunzip.on('data', (chunk: Buffer) => chunks.push(chunk));
      gunzip.on('end', () => {
        const json = Buffer.concat(chunks).toString('utf-8');
        resolve(JSON.parse(json));
      });
      gunzip.on('error', reject);
      gunzip.end(compressed);
    });
  } else {
    // Plain JSON
    const json = fs.readFileSync(resolvedPath, 'utf-8');
    return JSON.parse(json);
  }
}

/**
 * Save a strategy file with gzip compression
 */
async function saveStrategy(
  data: SerializedStrategy,
  filePath: string,
  quiet: boolean
): Promise<void> {
  const resolvedPath = path.resolve(filePath);
  const json = JSON.stringify(data);

  if (filePath.endsWith('.gz')) {
    // Compress with gzip
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(resolvedPath);
    await pipeline(Readable.from(json), gzip, output);
  } else {
    // Plain JSON (pretty-printed for debugging)
    fs.writeFileSync(resolvedPath, JSON.stringify(data, null, 2));
  }

  if (!quiet) {
    const stats = fs.statSync(resolvedPath);
    const sizeMB = stats.size / (1024 * 1024);
    if (sizeMB >= 1) {
      console.log(`Saved: ${resolvedPath} (${sizeMB.toFixed(1)} MB)`);
    } else {
      console.log(`Saved: ${resolvedPath} (${(stats.size / 1024).toFixed(1)} KB)`);
    }
  }
}

async function main() {
  const config = parseArgs();

  // Determine checkpoint path
  const checkpointPath = config.output.replace(/\.json(\.gz)?$/, '.checkpoint.json.gz');

  let trainer: MCCFRTrainer;
  let startIteration = 0;

  // Resume from checkpoint if specified
  if (config.resume) {
    if (!config.quiet) {
      console.log(`Resuming from: ${config.resume}`);
    }

    const data = await loadStrategy(config.resume);
    trainer = MCCFRTrainer.fromSerialized(data);
    startIteration = data.iterationsCompleted;

    if (!config.quiet) {
      console.log(`Loaded ${data.nodes.length.toLocaleString()} info sets`);
      console.log(`Continuing from iteration ${startIteration.toLocaleString()}`);
      console.log('');
    }
  } else {
    trainer = new MCCFRTrainer({
      iterations: config.iterations,
      seed: config.seed
    });
  }

  if (!config.quiet) {
    console.log('MCCFR Training for Texas 42');
    console.log('============================');
    console.log(`Target iterations: ${config.iterations.toLocaleString()}`);
    console.log(`Start iteration: ${startIteration.toLocaleString()}`);
    console.log(`Seed: ${config.seed}`);
    console.log(`Output: ${config.output}`);
    if (!config.noCheckpoint) {
      console.log(`Checkpoint interval: ${config.checkpointInterval.toLocaleString()}`);
    }
    console.log('');
  }

  const startTime = Date.now();
  let lastCheckpointIteration = startIteration;

  // Custom training loop with checkpointing
  const totalIterations = config.iterations;
  const progressInterval = config.progress ? 10 : 0;

  for (let i = startIteration; i < totalIterations; i++) {
    // Run one iteration
    trainer.runSingleIteration(i);

    // Progress output
    if (progressInterval > 0 && (i + 1) % progressInterval === 0 && !config.quiet) {
      const elapsed = (Date.now() - startTime) / 1000;
      const itersDone = i + 1 - startIteration;
      const rate = Math.round(itersDone / elapsed);
      const eta = rate > 0 ? Math.round((totalIterations - i - 1) / rate) : 0;
      const nodes = trainer.getRegretTable().size;

      process.stdout.write(
        `\rProgress: ${(i + 1).toLocaleString()}/${totalIterations.toLocaleString()} ` +
        `(${Math.round(100 * (i + 1) / totalIterations)}%) | ` +
        `${nodes.toLocaleString()} info sets | ` +
        `${rate.toLocaleString()} iter/s | ` +
        `ETA: ${eta}s   `
      );
    }

    // Checkpoint
    if (!config.noCheckpoint &&
        config.checkpointInterval > 0 &&
        (i + 1) % config.checkpointInterval === 0 &&
        i + 1 > lastCheckpointIteration) {

      const checkpoint = trainer.serializeCheckpoint(i + 1, config.seed, config.seed + totalIterations);
      await saveStrategy(checkpoint, checkpointPath, true); // quiet for checkpoints
      lastCheckpointIteration = i + 1;

      if (!config.quiet && !config.progress) {
        console.log(`Checkpoint saved at iteration ${(i + 1).toLocaleString()}`);
      }
    }
  }

  if (config.progress && !config.quiet) {
    process.stdout.write('\n');
  }

  const trainingTimeMs = Date.now() - startTime;
  const itersDone = totalIterations - startIteration;

  if (!config.quiet) {
    console.log('');
    console.log('Training complete!');
    console.log('------------------');
    console.log(`Iterations: ${totalIterations.toLocaleString()} (${itersDone.toLocaleString()} this run)`);
    console.log(`Info sets: ${trainer.getRegretTable().size.toLocaleString()}`);
    console.log(`Time: ${(trainingTimeMs / 1000).toFixed(1)}s`);
    console.log(`Rate: ${Math.round(itersDone / (trainingTimeMs / 1000)).toLocaleString()} iter/s`);
    console.log('');
  }

  // Save final output
  const serialized = trainer.serializeFinal(
    totalIterations,
    trainingTimeMs,
    config.seed,
    config.seed + totalIterations
  );

  await saveStrategy(serialized, config.output, config.quiet);

  // Remove checkpoint file on successful completion
  if (!config.noCheckpoint && fs.existsSync(checkpointPath)) {
    fs.unlinkSync(checkpointPath);
    if (!config.quiet) {
      console.log(`Removed checkpoint: ${checkpointPath}`);
    }
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
