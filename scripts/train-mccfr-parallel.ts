#!/usr/bin/env npx tsx
/**
 * Parallel MCCFR Training Script
 *
 * Spawns multiple child processes for true parallel training.
 * Each worker trains with a different seed, then results are merged.
 *
 * Usage:
 *   npx tsx scripts/train-mccfr-parallel.ts [options]
 *
 * Options:
 *   --workers <n>       Number of parallel workers (default: CPU cores - 1)
 *   --iterations <n>    Total iterations across all workers (default: 100000)
 *   --seed <n>          Base seed (workers use seed, seed+1M, ...) (default: 42)
 *   --output <file>     Output file (default: trained-strategy.json.gz)
 *   --quiet             Suppress output except errors
 */

import type { SerializedStrategy } from '../src/game/ai/cfr';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';
import { createGzip } from 'zlib';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface WorkerState {
  workerId: number;
  process: ChildProcess;
  iteration: number;
  totalIterations: number;
  infoSets: number;
  rate: number;
  status: 'running' | 'done' | 'error';
  outputFile: string;
}

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const cpuCount = os.cpus().length;

  const config = {
    workers: Math.max(1, cpuCount - 1),
    iterations: 100000,
    seed: 42,
    output: 'trained-strategy.json.gz',
    quiet: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case '--workers':
      case '-w':
        config.workers = parseInt(next ?? String(cpuCount), 10);
        i++;
        break;
      case '--iterations':
      case '-i':
        config.iterations = parseInt(next ?? '100000', 10);
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
      case '--quiet':
      case '-q':
        config.quiet = true;
        break;
      case '--help':
      case '-h':
        console.log(`
Parallel MCCFR Training Script (Multi-Process)

Usage:
  npx tsx scripts/train-mccfr-parallel.ts [options]

Options:
  --workers, -w <n>     Number of parallel workers (default: ${cpuCount - 1})
  --iterations, -i <n>  Total iterations across all workers (default: 100000)
  --seed, -s <n>        Base seed (workers use seed, seed+1M, ...) (default: 42)
  --output, -o <file>   Output file (default: trained-strategy.json.gz)
  --quiet, -q           Suppress output except errors
  --help, -h            Show this help message

Examples:
  npx tsx scripts/train-mccfr-parallel.ts --workers 4 --iterations 100000
  npx tsx scripts/train-mccfr-parallel.ts -w 8 -i 500000 -o big-strategy.json.gz
        `);
        process.exit(0);
    }
  }

  return config;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}

function renderProgress(workers: WorkerState[], startTime: number, totalIterations: number, quiet: boolean): void {
  if (quiet) return;

  const elapsed = Date.now() - startTime;
  const totalDone = workers.reduce((sum, w) => sum + w.iteration, 0);
  const totalRate = workers.reduce((sum, w) => sum + w.rate, 0);
  const totalInfoSets = workers.reduce((sum, w) => sum + w.infoSets, 0);
  const overallProgress = Math.round(100 * totalDone / totalIterations);
  const eta = totalRate > 0 ? Math.round((totalIterations - totalDone) / totalRate) : 0;
  const doneCount = workers.filter(w => w.status === 'done').length;

  // Clear and redraw
  process.stdout.write('\x1B[2J\x1B[H');

  console.log('╔════════════════════════════════════════════════════════════════════╗');
  console.log('║           MCCFR Parallel Training - Texas 42                       ║');
  console.log('╠════════════════════════════════════════════════════════════════════╣');
  console.log(`║  Progress: ${String(overallProgress).padStart(3)}%  │  ${totalDone.toLocaleString().padStart(8)} / ${totalIterations.toLocaleString().padEnd(8)}  │  Done: ${doneCount}/${workers.length}  ║`);
  console.log(`║  Rate: ${totalRate.toLocaleString().padStart(6)} iter/s  │  ETA: ${formatDuration(eta * 1000).padEnd(8)}  │  Info: ${totalInfoSets.toLocaleString().padStart(8)}  ║`);
  console.log(`║  Elapsed: ${formatDuration(elapsed).padEnd(10)}                                             ║`);
  console.log('╠════════════════════════════════════════════════════════════════════╣');

  for (const w of workers) {
    const pct = w.totalIterations > 0 ? Math.round(100 * w.iteration / w.totalIterations) : 0;
    const barLen = 20;
    const filled = Math.floor(pct / 5);
    const bar = '█'.repeat(filled) + '░'.repeat(barLen - filled);
    const icon = w.status === 'done' ? '✓' : w.status === 'error' ? '✗' : '●';
    console.log(`║  ${icon} W${String(w.workerId).padStart(2)}: [${bar}] ${String(pct).padStart(3)}%  │  ${w.rate.toLocaleString().padStart(4)}/s  │  ${w.infoSets.toLocaleString().padStart(6)} sets  ║`);
  }

  console.log('╚════════════════════════════════════════════════════════════════════╝');
}

function spawnWorker(
  workerId: number,
  iterations: number,
  seed: number,
  outputFile: string,
  onProgress: (workerId: number, iteration: number, total: number, infoSets: number, rate: number) => void,
  onDone: (workerId: number) => void,
  onError: (workerId: number, error: string) => void
): ChildProcess {
  const child = spawn('npx', [
    'tsx',
    path.join(__dirname, 'train-mccfr.ts'),
    '--iterations', String(iterations),
    '--seed', String(seed),
    '--output', outputFile,
    '--progress'
    // Note: NOT using --quiet so we can parse progress output
  ], {
    stdio: ['ignore', 'pipe', 'pipe']
  });

  child.stdout?.on('data', (data: Buffer) => {
    const text = data.toString();
    // Parse progress output: "Progress: X/Y (Z%) | N info sets | R iter/s | ETA: Ts"
    const match = text.match(/Progress:\s*([\d,]+)\/([\d,]+).*?\|\s*([\d,]+)\s*info sets\s*\|\s*([\d,]+)\s*iter\/s/);
    if (match) {
      const iteration = parseInt(match[1]!.replace(/,/g, ''), 10);
      const total = parseInt(match[2]!.replace(/,/g, ''), 10);
      const infoSets = parseInt(match[3]!.replace(/,/g, ''), 10);
      const rate = parseInt(match[4]!.replace(/,/g, ''), 10);
      onProgress(workerId, iteration, total, infoSets, rate);
    }
  });

  child.stderr?.on('data', (_data: Buffer) => {
    // Ignore stderr for now, could log errors
  });

  child.on('close', (code) => {
    if (code === 0) {
      onDone(workerId);
    } else {
      onError(workerId, `Exit code ${code}`);
    }
  });

  child.on('error', (err) => {
    onError(workerId, err.message);
  });

  return child;
}

/**
 * Load a strategy file (supports both .json and .json.gz)
 */
function loadStrategySync(filePath: string): SerializedStrategy {
  if (filePath.endsWith('.gz')) {
    // Decompress gzipped file synchronously using zlib
    const { gunzipSync } = require('zlib');
    const compressed = fs.readFileSync(filePath);
    const json = gunzipSync(compressed).toString('utf-8');
    return JSON.parse(json);
  } else {
    // Plain JSON
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  }
}

/**
 * Save a strategy file with optional gzip compression
 */
async function saveStrategy(
  data: SerializedStrategy,
  filePath: string
): Promise<void> {
  const json = JSON.stringify(data);

  if (filePath.endsWith('.gz')) {
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(filePath);
    await pipeline(Readable.from(json), gzip, output);
  } else {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
  }
}

/**
 * Streaming merge - processes one file at a time to avoid OOM.
 * Uses raw Maps instead of RegretTable to minimize memory overhead.
 * Supports both .json and .json.gz input files.
 */
function mergeStrategies(files: string[], quiet: boolean): SerializedStrategy {
  // Merged data stored as compact Maps
  // Map<infoSetKey, { regrets: Map<action, number>, stratSum: Map<action, number>, visits: number }>
  const merged = new Map<string, {
    regrets: Map<string, number>;
    stratSum: Map<string, number>;
    visits: number;
  }>();

  let totalIterations = 0;
  let maxTrainingTime = 0;
  let baseConfig: SerializedStrategy['config'] | null = null;
  let filesProcessed = 0;

  for (const file of files) {
    if (!fs.existsSync(file)) continue;

    filesProcessed++;
    if (!quiet) {
      process.stdout.write(`\rMerging file ${filesProcessed}/${files.length}...`);
    }

    // Read and parse one file at a time (supports .gz)
    const data = loadStrategySync(file);

    if (!baseConfig) {
      baseConfig = data.config;
    }

    totalIterations += data.iterationsCompleted;
    maxTrainingTime = Math.max(maxTrainingTime, data.trainingTimeMs);

    // Merge nodes directly from serialized format
    // Format: nodes[].regrets is Array<[ActionKey, number]>
    for (const node of data.nodes) {
      let mergedNode = merged.get(node.key);
      if (!mergedNode) {
        mergedNode = {
          regrets: new Map(),
          stratSum: new Map(),
          visits: 0
        };
        merged.set(node.key, mergedNode);
      }

      // Merge regrets - format is Array<[action, value]>
      for (const [action, regret] of node.regrets) {
        const current = mergedNode.regrets.get(action) ?? 0;
        mergedNode.regrets.set(action, current + regret);
      }

      // Merge strategy sums - format is Array<[action, value]>
      for (const [action, stratSum] of node.strategySum) {
        const current = mergedNode.stratSum.get(action) ?? 0;
        mergedNode.stratSum.set(action, current + stratSum);
      }

      mergedNode.visits += node.visitCount;
    }
  }

  if (!baseConfig) {
    throw new Error('No results to merge');
  }

  if (!quiet) {
    console.log(`\nConverting ${merged.size.toLocaleString()} info sets to output format...`);
  }

  // Convert to serialized format: Array<[ActionKey, number]>
  const nodes: SerializedStrategy['nodes'] = [];
  for (const [key, node] of merged) {
    nodes.push({
      key,
      regrets: Array.from(node.regrets.entries()),
      strategySum: Array.from(node.stratSum.entries()),
      visitCount: node.visits
    });
  }

  return {
    version: 1,
    config: { ...baseConfig, iterations: totalIterations },
    iterationsCompleted: totalIterations,
    trainingTimeMs: maxTrainingTime,
    trainedAt: new Date().toISOString(),
    nodes
  };
}

async function main() {
  const config = parseArgs();
  const iterationsPerWorker = Math.ceil(config.iterations / config.workers);

  // Create temp directory for worker outputs
  const tempDir = path.join(os.tmpdir(), `mccfr-${Date.now()}`);
  fs.mkdirSync(tempDir, { recursive: true });

  if (!config.quiet) {
    console.log('Parallel MCCFR Training for Texas 42');
    console.log('=====================================');
    console.log(`Workers: ${config.workers} (separate processes)`);
    console.log(`Total iterations: ${config.iterations.toLocaleString()}`);
    console.log(`Per worker: ${iterationsPerWorker.toLocaleString()}`);
    console.log(`Base seed: ${config.seed}`);
    console.log(`Output: ${config.output}`);
    console.log('');
    console.log('Spawning workers...');
  }

  const startTime = Date.now();

  // Initialize worker states
  const workers: WorkerState[] = [];
  const outputFiles: string[] = [];

  for (let i = 0; i < config.workers; i++) {
    const outputFile = path.join(tempDir, `worker-${i}.json`);
    outputFiles.push(outputFile);

    workers.push({
      workerId: i,
      process: null as unknown as ChildProcess,
      iteration: 0,
      totalIterations: iterationsPerWorker,
      infoSets: 0,
      rate: 0,
      status: 'running',
      outputFile
    });
  }

  // Track completion
  let completedCount = 0;
  const allDone = new Promise<void>((resolve) => {
    const checkDone = () => {
      if (completedCount >= config.workers) {
        resolve();
      }
    };

    // Spawn workers
    for (let i = 0; i < config.workers; i++) {
      const workerSeed = config.seed + i * 1000000;

      workers[i]!.process = spawnWorker(
        i,
        iterationsPerWorker,
        workerSeed,
        outputFiles[i]!,
        // onProgress
        (workerId, iteration, total, infoSets, rate) => {
          workers[workerId]!.iteration = iteration;
          workers[workerId]!.totalIterations = total;
          workers[workerId]!.infoSets = infoSets;
          workers[workerId]!.rate = rate;
          renderProgress(workers, startTime, config.iterations, config.quiet);
        },
        // onDone
        (workerId) => {
          workers[workerId]!.status = 'done';
          workers[workerId]!.iteration = workers[workerId]!.totalIterations;
          workers[workerId]!.rate = 0;
          completedCount++;
          renderProgress(workers, startTime, config.iterations, config.quiet);
          checkDone();
        },
        // onError
        (workerId, error) => {
          workers[workerId]!.status = 'error';
          workers[workerId]!.rate = 0;
          completedCount++;
          if (!config.quiet) {
            console.error(`Worker ${workerId} error: ${error}`);
          }
          checkDone();
        }
      );
    }
  });

  // Initial render
  renderProgress(workers, startTime, config.iterations, config.quiet);

  // Wait for all workers
  await allDone;

  // Final render
  renderProgress(workers, startTime, config.iterations, config.quiet);

  if (!config.quiet) {
    console.log('');
    console.log('Merging results...');
  }

  // Merge results
  const merged = mergeStrategies(outputFiles, config.quiet);

  // Save merged result
  const outputPath = path.resolve(config.output);

  if (!config.quiet) {
    console.log('Saving merged strategy...');
  }

  await saveStrategy(merged, outputPath);

  // Cleanup temp files
  for (const file of outputFiles) {
    try { fs.unlinkSync(file); } catch {}
  }
  try { fs.rmdirSync(tempDir); } catch {}

  const elapsed = Date.now() - startTime;

  if (!config.quiet) {
    const stats = fs.statSync(outputPath);
    const sizeMB = stats.size / (1024 * 1024);
    console.log('');
    console.log('Training Complete!');
    console.log('==================');
    console.log(`Total iterations: ${merged.iterationsCompleted.toLocaleString()}`);
    console.log(`Unique info sets: ${merged.nodes.length.toLocaleString()}`);
    console.log(`Total time: ${formatDuration(elapsed)}`);
    console.log(`Effective rate: ${Math.round(merged.iterationsCompleted / (elapsed / 1000)).toLocaleString()} iter/s`);
    console.log(`Output: ${outputPath}`);
    if (sizeMB >= 1) {
      console.log(`Size: ${sizeMB.toFixed(1)} MB`);
    } else {
      console.log(`Size: ${(stats.size / 1024).toFixed(1)} KB`);
    }
  }
}

main().catch(err => {
  console.error('Training failed:', err);
  process.exit(1);
});
