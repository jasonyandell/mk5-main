#!/usr/bin/env npx tsx
/**
 * MCCFR Strategy Merge Tool
 *
 * Merges multiple trained MCCFR strategies into one.
 * Useful for combining results from partitioned/distributed training runs.
 *
 * Usage:
 *   npx tsx scripts/merge-strategies.ts --inputs file1.json.gz file2.json.gz -o combined.json.gz
 *
 * Options:
 *   --inputs <files...>  Input strategy files to merge (supports .json and .json.gz)
 *   --output, -o <file>  Output file (default: merged-strategy.json.gz)
 *   --quiet, -q          Suppress output except errors
 */

import type { SerializedStrategy } from '../src/game/ai/cfr';
import * as fs from 'fs';
import * as path from 'path';
import { createGzip, gunzipSync } from 'zlib';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    inputs: [] as string[],
    output: 'merged-strategy.json.gz',
    quiet: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case '--inputs':
      case '-i':
        // Collect all following arguments until next flag
        while (i + 1 < args.length && !args[i + 1]!.startsWith('-')) {
          config.inputs.push(args[++i]!);
        }
        break;
      case '--output':
      case '-o':
        config.output = next ?? 'merged-strategy.json.gz';
        i++;
        break;
      case '--quiet':
      case '-q':
        config.quiet = true;
        break;
      case '--help':
      case '-h':
        console.log(`
MCCFR Strategy Merge Tool

Usage:
  npx tsx scripts/merge-strategies.ts --inputs file1.json.gz file2.json.gz -o combined.json.gz

Options:
  --inputs, -i <files...>  Input strategy files to merge (supports .json and .json.gz)
  --output, -o <file>      Output file (default: merged-strategy.json.gz)
  --quiet, -q              Suppress output except errors
  --help, -h               Show this help message

Examples:
  # Merge two partitioned training runs
  npx tsx scripts/merge-strategies.ts --inputs part1.json.gz part2.json.gz -o combined.json.gz

  # Merge multiple files with glob (shell expansion)
  npx tsx scripts/merge-strategies.ts --inputs training-*.json.gz -o full-strategy.json.gz

Notes:
  - Regrets and strategy sums are added together (correct for partitioned training)
  - Visit counts are summed
  - All input files must use the same configuration version
        `);
        process.exit(0);
      default:
        // Positional arguments are also treated as inputs if no flag
        if (arg && !arg.startsWith('-')) {
          config.inputs.push(arg);
        }
    }
  }

  return config;
}

/**
 * Load a strategy file (supports both .json and .json.gz)
 */
function loadStrategy(filePath: string): SerializedStrategy {
  const resolvedPath = path.resolve(filePath);

  if (filePath.endsWith('.gz')) {
    const compressed = fs.readFileSync(resolvedPath);
    const json = gunzipSync(compressed).toString('utf-8');
    return JSON.parse(json);
  } else {
    return JSON.parse(fs.readFileSync(resolvedPath, 'utf-8'));
  }
}

/**
 * Save a strategy file with optional gzip compression
 */
async function saveStrategy(data: SerializedStrategy, filePath: string): Promise<void> {
  const resolvedPath = path.resolve(filePath);
  const json = JSON.stringify(data);

  if (filePath.endsWith('.gz')) {
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(resolvedPath);
    await pipeline(Readable.from(json), gzip, output);
  } else {
    fs.writeFileSync(resolvedPath, JSON.stringify(data, null, 2));
  }
}

/**
 * Streaming merge - processes one file at a time to avoid OOM.
 */
function mergeStrategies(
  files: string[],
  quiet: boolean
): SerializedStrategy {
  const merged = new Map<string, {
    regrets: Map<string, number>;
    stratSum: Map<string, number>;
    visits: number;
  }>();

  let totalIterations = 0;
  let maxTrainingTime = 0;
  let baseConfig: SerializedStrategy['config'] | null = null;
  let minSeedStart = Infinity;
  let maxSeedEnd = -Infinity;
  let filesProcessed = 0;

  for (const file of files) {
    if (!fs.existsSync(file)) {
      console.error(`Warning: File not found: ${file}`);
      continue;
    }

    filesProcessed++;
    if (!quiet) {
      process.stdout.write(`\rMerging file ${filesProcessed}/${files.length}: ${path.basename(file)}...`);
    }

    const data = loadStrategy(file);

    if (!baseConfig) {
      baseConfig = data.config;
    }

    // Validate version compatibility
    if (data.version !== 1) {
      throw new Error(`Unsupported version ${data.version} in ${file}`);
    }

    totalIterations += data.iterationsCompleted;
    maxTrainingTime = Math.max(maxTrainingTime, data.trainingTimeMs);

    // Track seed ranges
    if (data.seedStart !== undefined) {
      minSeedStart = Math.min(minSeedStart, data.seedStart);
    }
    if (data.seedEnd !== undefined) {
      maxSeedEnd = Math.max(maxSeedEnd, data.seedEnd);
    }

    // Merge nodes
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

      for (const [action, regret] of node.regrets) {
        const current = mergedNode.regrets.get(action) ?? 0;
        mergedNode.regrets.set(action, current + regret);
      }

      for (const [action, stratSum] of node.strategySum) {
        const current = mergedNode.stratSum.get(action) ?? 0;
        mergedNode.stratSum.set(action, current + stratSum);
      }

      mergedNode.visits += node.visitCount;
    }
  }

  if (!baseConfig) {
    throw new Error('No valid input files to merge');
  }

  if (!quiet) {
    console.log(`\nConverting ${merged.size.toLocaleString()} info sets to output format...`);
  }

  const nodes: SerializedStrategy['nodes'] = [];
  for (const [key, node] of merged) {
    nodes.push({
      key,
      regrets: Array.from(node.regrets.entries()),
      strategySum: Array.from(node.stratSum.entries()),
      visitCount: node.visits
    });
  }

  const result: SerializedStrategy = {
    version: 1,
    config: { ...baseConfig, iterations: totalIterations },
    iterationsCompleted: totalIterations,
    trainingTimeMs: maxTrainingTime,
    trainedAt: new Date().toISOString(),
    nodes,
    isCheckpoint: false
  };

  // Only set seed range if we have valid values
  if (minSeedStart !== Infinity) {
    result.seedStart = minSeedStart;
  }
  if (maxSeedEnd !== -Infinity) {
    result.seedEnd = maxSeedEnd;
  }

  return result;
}

async function main() {
  const config = parseArgs();

  if (config.inputs.length === 0) {
    console.error('Error: No input files specified');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  if (!config.quiet) {
    console.log('MCCFR Strategy Merge Tool');
    console.log('=========================');
    console.log(`Input files: ${config.inputs.length}`);
    console.log(`Output: ${config.output}`);
    console.log('');
  }

  const startTime = Date.now();

  // Merge all strategies
  const merged = mergeStrategies(config.inputs, config.quiet);

  // Save merged result
  const outputPath = path.resolve(config.output);
  await saveStrategy(merged, outputPath);

  const elapsed = Date.now() - startTime;

  if (!config.quiet) {
    const stats = fs.statSync(outputPath);
    const sizeMB = stats.size / (1024 * 1024);

    console.log('');
    console.log('Merge Complete!');
    console.log('===============');
    console.log(`Total iterations: ${merged.iterationsCompleted.toLocaleString()}`);
    console.log(`Unique info sets: ${merged.nodes.length.toLocaleString()}`);
    console.log(`Merge time: ${(elapsed / 1000).toFixed(1)}s`);
    console.log(`Output: ${outputPath}`);
    if (sizeMB >= 1) {
      console.log(`Size: ${sizeMB.toFixed(1)} MB`);
    } else {
      console.log(`Size: ${(stats.size / 1024).toFixed(1)} KB`);
    }
    if (merged.seedStart !== undefined && merged.seedEnd !== undefined) {
      console.log(`Seed range: ${merged.seedStart} - ${merged.seedEnd}`);
    }
  }
}

main().catch(err => {
  console.error('Merge failed:', err);
  process.exit(1);
});
