#!/usr/bin/env npx tsx
/**
 * Collect CFR tractability metrics for Texas 42.
 *
 * Usage:
 *   npx tsx scripts/collect-cfr-metrics.ts [numGames] [seed]
 *
 * Examples:
 *   npx tsx scripts/collect-cfr-metrics.ts 100      # Quick test
 *   npx tsx scripts/collect-cfr-metrics.ts 1000    # Better sample
 *   npx tsx scripts/collect-cfr-metrics.ts 10000   # Full analysis
 */

import { runMetricsCollection, generateReport } from '../src/game/ai/cfr-metrics';

async function main() {
  const numGames = parseInt(process.argv[2] || '100', 10);
  const seed = parseInt(process.argv[3] || '12345', 10);

  console.log('═'.repeat(70));
  console.log('CFR TRACTABILITY METRICS FOR TEXAS 42');
  console.log('═'.repeat(70));
  console.log(`Games to simulate: ${numGames}`);
  console.log(`Base seed: ${seed}`);
  console.log('');

  const startTime = Date.now();

  try {
    const metrics = await runMetricsCollection(numGames, seed, (game, total) => {
      const elapsed = (Date.now() - startTime) / 1000;
      const pct = ((game / total) * 100).toFixed(0);
      process.stdout.write(`\rGame ${game}/${total} (${pct}%) - ${elapsed.toFixed(1)}s elapsed`);
    });
    process.stdout.write('\r' + ' '.repeat(60) + '\r');

    const elapsed = (Date.now() - startTime) / 1000;
    const report = generateReport(metrics, numGames);

    console.log('─'.repeat(70));
    console.log('RESULTS');
    console.log('─'.repeat(70));
    console.log(`Time: ${elapsed.toFixed(2)}s (${(numGames / elapsed).toFixed(1)} games/sec)`);
    console.log(`Games simulated: ${report.gamesSimulated}`);
    console.log(`Total decision points: ${report.totalDecisionPoints.toLocaleString()}`);
    console.log('');

    console.log('BRANCHING FACTOR (legal actions per decision)');
    console.log('─'.repeat(40));
    console.log(`  Min: ${report.branchingFactor.min}`);
    console.log(`  Max: ${report.branchingFactor.max}`);
    console.log(`  Mean: ${report.branchingFactor.mean}`);
    console.log(`  Median: ${report.branchingFactor.median}`);
    console.log('');
    console.log('  Distribution:');
    const sortedBF = Object.entries(report.branchingFactor.distribution)
      .map(([k, v]) => [parseInt(k), v] as [number, number])
      .sort((a, b) => a[0] - b[0]);
    for (const [bf, count] of sortedBF) {
      const pct = ((count / report.totalDecisionPoints) * 100).toFixed(1);
      const bar = '█'.repeat(Math.min(50, Math.round(parseFloat(pct))));
      console.log(`    ${bf.toString().padStart(2)}: ${count.toString().padStart(6)} (${pct.padStart(5)}%) ${bar}`);
    }
    console.log('');

    console.log('═'.repeat(70));
    console.log('INFORMATION SET COMPARISON');
    console.log('═'.repeat(70));
    console.log('');
    console.log('                    │  Raw State  │  Canonical  │ Count-Centric');
    console.log('────────────────────┼─────────────┼─────────────┼──────────────');
    console.log(`  Unique states     │ ${report.uniqueInfoSets.toLocaleString().padStart(11)} │ ${report.canonicalInfoSets.toLocaleString().padStart(11)} │ ${report.countCentricInfoSets.toLocaleString().padStart(12)}`);
    console.log(`  Compression ratio │      1.00x  │ ${report.compressionRatio.toFixed(2).padStart(10)}x │ ${report.countCentricCompressionRatio.toFixed(2).padStart(11)}x`);
    console.log(`  Revisitation rate │       n/a   │ ${report.revisitationRate.toFixed(2).padStart(11)} │ ${report.countCentricRevisitationRate.toFixed(2).padStart(12)}`);
    console.log(`  Singleton rate    │       n/a   │ ${report.singletonRate.toFixed(1).padStart(10)}% │ ${report.countCentricSingletonRate.toFixed(1).padStart(11)}%`);
    console.log('');

    // Total compression from raw to count-centric
    const totalCompression = report.uniqueInfoSets / report.countCentricInfoSets;
    console.log(`  TOTAL COMPRESSION (raw → count-centric): ${totalCompression.toFixed(1)}x`);
    console.log('');

    console.log('═'.repeat(70));
    console.log('TRACTABILITY ASSESSMENT');
    console.log('═'.repeat(70));
    console.log('');

    // Extrapolate to full game tree
    const decisionPointsPerGame = report.totalDecisionPoints / numGames;
    console.log(`  Decision points per game: ~${decisionPointsPerGame.toFixed(1)}`);

    const canonicalGrowthRate = report.canonicalInfoSets / numGames;
    const countCentricGrowthRate = report.countCentricInfoSets / numGames;
    console.log(`  New canonical states per game: ~${canonicalGrowthRate.toFixed(1)}`);
    console.log(`  New count-centric buckets per game: ~${countCentricGrowthRate.toFixed(1)}`);
    console.log('');

    // Assessment
    if (report.countCentricInfoSets < 100000) {
      console.log('  ✓  COUNT-CENTRIC ABSTRACTION IS HIGHLY EFFECTIVE');
      console.log(`     Only ${report.countCentricInfoSets.toLocaleString()} buckets vs ${report.canonicalInfoSets.toLocaleString()} canonical states`);
      console.log('     CFR on count-centric buckets should be tractable!');
    } else if (report.countCentricInfoSets < 1000000) {
      console.log('  ⚡ COUNT-CENTRIC ABSTRACTION SHOWS PROMISE');
      console.log(`     ${report.countCentricInfoSets.toLocaleString()} buckets is manageable`);
      console.log('     Further refinement may help');
    } else {
      console.log('  ⚠️  COUNT-CENTRIC ABSTRACTION INSUFFICIENT');
      console.log(`     ${report.countCentricInfoSets.toLocaleString()} buckets still large`);
      console.log('     Need additional abstraction (hand strength bucketing?)');
    }

    if (report.countCentricSingletonRate < 50) {
      console.log('');
      console.log('  ✓  LOW SINGLETON RATE: Good state revisitation');
      console.log(`     Only ${report.countCentricSingletonRate.toFixed(1)}% of count-centric states are unique`);
    }

    console.log('');
    console.log('─'.repeat(70));
    console.log(`Current: ${numGames} games → ${report.countCentricInfoSets.toLocaleString()} count-centric buckets`);

  } catch (error) {
    console.error('\nError during collection:', error);
    process.exit(1);
  }
}

main();
