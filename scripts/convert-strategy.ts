#!/usr/bin/env npx tsx
/**
 * Convert MCCFR Strategy Format
 *
 * Converts between JSON (.json.gz) and compact binary (.cfr.gz/.cfr-deploy.gz/.cfd2.gz) formats.
 *
 * Usage:
 *   npx tsx scripts/convert-strategy.ts --input strategy.json.gz --output strategy.cfr.gz
 *   npx tsx scripts/convert-strategy.ts --input strategy.cfr.gz --output strategy.json.gz
 *   npx tsx scripts/convert-strategy.ts --input strategy.json.gz --output strategy.cfd2.gz
 *
 * Options:
 *   --input, -i <file>   Input file (.json.gz, .cfr.gz, .cfr-deploy.gz, or .cfd2.gz)
 *   --output, -o <file>  Output file (.json.gz, .cfr.gz, .cfr-deploy.gz, or .cfd2.gz)
 *   --stats              Show size statistics
 *   --verify             Verify round-trip conversion
 *   --quiet, -q          Suppress output
 */

import type { SerializedStrategy } from '../src/game/ai/cfr/types';
import { serializeCompact, deserializeCompact, serializeCompactDeploy } from '../src/game/ai/cfr/compact-format';
import { serializeCFD2, deserializeCFD2, detectFormat } from '../src/game/ai/cfr/compact-format-v2';
import * as fs from 'fs';
import * as path from 'path';
import { createGzip, gunzipSync } from 'zlib';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

// Parse arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    input: '',
    output: '',
    stats: false,
    verify: false,
    quiet: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case '--input':
      case '-i':
        config.input = next || '';
        i++;
        break;
      case '--output':
      case '-o':
        config.output = next || '';
        i++;
        break;
      case '--stats':
        config.stats = true;
        break;
      case '--verify':
        config.verify = true;
        break;
      case '--quiet':
      case '-q':
        config.quiet = true;
        break;
      case '--help':
      case '-h':
        console.log(`
Convert MCCFR Strategy Format

Usage:
  npx tsx scripts/convert-strategy.ts --input strategy.json.gz --output strategy.cfr.gz
  npx tsx scripts/convert-strategy.ts --input strategy.cfr.gz --output strategy.json.gz
  npx tsx scripts/convert-strategy.ts --input strategy.json.gz --output strategy.cfd2.gz

Options:
  --input, -i <file>   Input file (.json.gz, .cfr.gz, .cfr-deploy.gz, or .cfd2.gz)
  --output, -o <file>  Output file (.json.gz, .cfr.gz, .cfr-deploy.gz, or .cfd2.gz)
  --stats              Show size statistics
  --verify             Verify round-trip conversion
  --quiet, -q          Suppress output
  --help, -h           Show this help message

Supported formats:
  .json.gz       - JSON format with gzip compression (full fidelity, largest)
  .cfr.gz        - Compact binary format with regrets (for continued training)
  .cfr-deploy.gz - CFD1 deploy format, strategy only (for mobile apps)
  .cfd2.gz       - CFD2 ultra-compact format (smallest, best for mobile)
`);
        process.exit(0);
    }
  }

  return config;
}

function isJsonFormat(file: string): boolean {
  return file.endsWith('.json.gz') || file.endsWith('.json');
}

function isCompactFormat(file: string): boolean {
  return file.endsWith('.cfr.gz') || file.endsWith('.cfr');
}

function isDeployFormat(file: string): boolean {
  return file.endsWith('.cfr-deploy.gz') || file.endsWith('.cfr-deploy');
}

function isV2Format(file: string): boolean {
  return file.endsWith('.cfd2.gz') || file.endsWith('.cfd2');
}

function loadJsonStrategy(filePath: string): SerializedStrategy {
  const content = fs.readFileSync(filePath);
  if (filePath.endsWith('.gz')) {
    return JSON.parse(gunzipSync(content).toString());
  }
  return JSON.parse(content.toString());
}

function loadCompactStrategy(filePath: string): SerializedStrategy {
  let buffer: Buffer;
  if (filePath.endsWith('.gz')) {
    buffer = gunzipSync(fs.readFileSync(filePath));
  } else {
    buffer = fs.readFileSync(filePath);
  }

  // Auto-detect format version
  const format = detectFormat(buffer);
  if (format === 'CFD2') {
    return deserializeCFD2(buffer);
  }
  return deserializeCompact(buffer);
}

function loadV2Strategy(filePath: string): SerializedStrategy {
  let buffer: Buffer;
  if (filePath.endsWith('.gz')) {
    buffer = gunzipSync(fs.readFileSync(filePath));
  } else {
    buffer = fs.readFileSync(filePath);
  }
  return deserializeCFD2(buffer);
}

async function saveJsonStrategy(strategy: SerializedStrategy, filePath: string): Promise<void> {
  const json = JSON.stringify(strategy);
  if (filePath.endsWith('.gz')) {
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(filePath);
    await pipeline(Readable.from(json), gzip, output);
  } else {
    fs.writeFileSync(filePath, json);
  }
}

async function saveCompactStrategy(strategy: SerializedStrategy, filePath: string): Promise<void> {
  const buffer = serializeCompact(strategy);
  if (filePath.endsWith('.gz')) {
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(filePath);
    await pipeline(Readable.from(buffer), gzip, output);
  } else {
    fs.writeFileSync(filePath, buffer);
  }
}

async function saveDeployStrategy(strategy: SerializedStrategy, filePath: string): Promise<void> {
  const buffer = serializeCompactDeploy(strategy);
  if (filePath.endsWith('.gz')) {
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(filePath);
    await pipeline(Readable.from(buffer), gzip, output);
  } else {
    fs.writeFileSync(filePath, buffer);
  }
}

async function saveV2Strategy(strategy: SerializedStrategy, filePath: string): Promise<void> {
  const buffer = serializeCFD2(strategy);
  if (filePath.endsWith('.gz')) {
    const gzip = createGzip({ level: 6 });
    const output = fs.createWriteStream(filePath);
    await pipeline(Readable.from(buffer), gzip, output);
  } else {
    fs.writeFileSync(filePath, buffer);
  }
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  } else if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${bytes} bytes`;
}

async function main() {
  const config = parseArgs();

  if (!config.input) {
    console.error('Error: --input is required');
    process.exit(1);
  }

  // If just stats, show input info
  if (config.stats && !config.output) {
    const inputPath = path.resolve(config.input);
    const inputSize = fs.statSync(inputPath).size;

    let strategy: SerializedStrategy;
    if (isJsonFormat(config.input)) {
      strategy = loadJsonStrategy(inputPath);
    } else if (isV2Format(config.input)) {
      strategy = loadV2Strategy(inputPath);
    } else if (isCompactFormat(config.input) || isDeployFormat(config.input)) {
      strategy = loadCompactStrategy(inputPath);
    } else {
      console.error('Unknown input format');
      process.exit(1);
    }

    console.log('Strategy Statistics');
    console.log('===================');
    console.log(`File: ${config.input}`);
    console.log(`Size: ${formatSize(inputSize)}`);
    console.log(`Nodes: ${strategy.nodes.length.toLocaleString()}`);
    console.log(`Iterations: ${strategy.iterationsCompleted.toLocaleString()}`);

    // Calculate total actions
    let totalActions = 0;
    for (const node of strategy.nodes) {
      totalActions += node.regrets.length;
    }
    console.log(`Total actions: ${totalActions.toLocaleString()}`);
    console.log(`Avg actions/node: ${(totalActions / strategy.nodes.length).toFixed(2)}`);

    // Estimate compact sizes
    if (isJsonFormat(config.input)) {
      const compactBuffer = serializeCompact(strategy);
      const deployBuffer = serializeCompactDeploy(strategy);
      const v2Buffer = serializeCFD2(strategy);
      console.log(`\nEstimated sizes (uncompressed):`);
      console.log(`  Compact (cfr):    ${formatSize(compactBuffer.length)} (${(inputSize / compactBuffer.length).toFixed(1)}x vs gzipped JSON)`);
      console.log(`  Deploy (CFD1):    ${formatSize(deployBuffer.length)} (${(inputSize / deployBuffer.length).toFixed(1)}x vs gzipped JSON)`);
      console.log(`  Ultra (CFD2):     ${formatSize(v2Buffer.length)} (${(inputSize / v2Buffer.length).toFixed(1)}x vs gzipped JSON)`);
      console.log(`  CFD2 vs CFD1:     ${((1 - v2Buffer.length / deployBuffer.length) * 100).toFixed(1)}% smaller`);
    }

    return;
  }

  if (!config.output) {
    console.error('Error: --output is required');
    process.exit(1);
  }

  const inputPath = path.resolve(config.input);
  const outputPath = path.resolve(config.output);

  if (!fs.existsSync(inputPath)) {
    console.error(`Error: Input file not found: ${inputPath}`);
    process.exit(1);
  }

  // Load input
  let strategy: SerializedStrategy;
  const inputSize = fs.statSync(inputPath).size;

  if (!config.quiet) {
    console.log(`Loading: ${config.input} (${formatSize(inputSize)})`);
  }

  if (isJsonFormat(config.input)) {
    strategy = loadJsonStrategy(inputPath);
  } else if (isV2Format(config.input)) {
    strategy = loadV2Strategy(inputPath);
  } else if (isCompactFormat(config.input) || isDeployFormat(config.input)) {
    strategy = loadCompactStrategy(inputPath);
  } else {
    console.error('Unknown input format. Use .json.gz, .cfr.gz, .cfr-deploy.gz, or .cfd2.gz');
    process.exit(1);
  }

  if (!config.quiet) {
    console.log(`Nodes: ${strategy.nodes.length.toLocaleString()}`);
  }

  // Save output
  if (isJsonFormat(config.output)) {
    await saveJsonStrategy(strategy, outputPath);
  } else if (isV2Format(config.output)) {
    await saveV2Strategy(strategy, outputPath);
  } else if (isDeployFormat(config.output)) {
    await saveDeployStrategy(strategy, outputPath);
  } else if (isCompactFormat(config.output)) {
    await saveCompactStrategy(strategy, outputPath);
  } else {
    console.error('Unknown output format. Use .json.gz, .cfr.gz, .cfr-deploy.gz, or .cfd2.gz');
    process.exit(1);
  }

  const outputSize = fs.statSync(outputPath).size;

  if (!config.quiet) {
    console.log(`Saved: ${config.output} (${formatSize(outputSize)})`);
    console.log(`Size change: ${formatSize(inputSize)} → ${formatSize(outputSize)} (${(inputSize / outputSize).toFixed(1)}x)`);
  }

  // Verify round-trip if requested
  if (config.verify) {
    if (!config.quiet) {
      console.log('\nVerifying round-trip...');
    }

    let reloaded: SerializedStrategy;
    if (isJsonFormat(config.output)) {
      reloaded = loadJsonStrategy(outputPath);
    } else if (isV2Format(config.output)) {
      reloaded = loadV2Strategy(outputPath);
    } else {
      reloaded = loadCompactStrategy(outputPath);
    }

    // Compare node counts (only nodes with data)
    const originalValidNodes = strategy.nodes.filter(n => n.strategySum.length > 0).length;
    const reloadedValidNodes = reloaded.nodes.filter(n => n.strategySum.length > 0).length;
    if (reloadedValidNodes !== originalValidNodes) {
      console.error(`Node count mismatch: ${originalValidNodes} → ${reloadedValidNodes}`);
      process.exit(1);
    }
    if (!config.quiet) {
      console.log(`Valid nodes: ${originalValidNodes.toLocaleString()}`);
    }

    // Compare iterations
    if (reloaded.iterationsCompleted !== strategy.iterationsCompleted) {
      console.error(`Iteration count mismatch: ${strategy.iterationsCompleted} → ${reloaded.iterationsCompleted}`);
      process.exit(1);
    }

    if (!config.quiet) {
      console.log('Verification passed!');
    }
  }
}

main().catch(err => {
  console.error('Conversion failed:', err);
  process.exit(1);
});
