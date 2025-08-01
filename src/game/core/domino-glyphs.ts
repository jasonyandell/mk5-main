import type { Domino } from '../types';

// Map of domino values to horizontal Unicode glyphs
const HORIZONTAL_DOMINO_GLYPHS: Record<string, string> = {
  '0-0': '🀰', '0-1': '🀱', '0-2': '🀲', '0-3': '🀳', '0-4': '🀴', '0-5': '🀵', '0-6': '🀶',
  '1-1': '🀷', '1-2': '🀸', '1-3': '🀹', '1-4': '🀺', '1-5': '🀻', '1-6': '🀼',
  '2-2': '🀽', '2-3': '🀾', '2-4': '🀿', '2-5': '🁀', '2-6': '🁁',
  '3-3': '🁂', '3-4': '🁃', '3-5': '🁄', '3-6': '🁅',
  '4-4': '🁆', '4-5': '🁇', '4-6': '🁈',
  '5-5': '🁉', '5-6': '🁊',
  '6-6': '🁋'
};

// Map of domino values to vertical Unicode glyphs
const VERTICAL_DOMINO_GLYPHS: Record<string, string> = {
  '0-0': '🁌', '0-1': '🁍', '0-2': '🁎', '0-3': '🁏', '0-4': '🁐', '0-5': '🁑', '0-6': '🁒',
  '1-1': '🁓', '1-2': '🁔', '1-3': '🁕', '1-4': '🁖', '1-5': '🁗', '1-6': '🁘',
  '2-2': '🁙', '2-3': '🁚', '2-4': '🁛', '2-5': '🁜', '2-6': '🁝',
  '3-3': '🁞', '3-4': '🁟', '3-5': '🁠', '3-6': '🁡',
  '4-4': '🁢', '4-5': '🁣', '4-6': '🁤',
  '5-5': '🁥', '5-6': '🁦',
  '6-6': '🁧'
};

// Special domino glyphs
export const DOMINO_BACK_HORIZONTAL = '🁨';
export const DOMINO_BACK_VERTICAL = '🁩';

/**
 * Convert a domino to its Unicode glyph representation
 * @param domino The domino to convert
 * @param vertical Whether to use vertical orientation (default: false)
 * @returns Unicode domino glyph or fallback text representation
 */
export function dominoToGlyph(domino: Domino, vertical = false): string {
  // Normalize the key to always have lower value first
  const low = Math.min(domino.low, domino.high);
  const high = Math.max(domino.low, domino.high);
  const key = `${low}-${high}`;
  
  const glyphMap = vertical ? VERTICAL_DOMINO_GLYPHS : HORIZONTAL_DOMINO_GLYPHS;
  return glyphMap[key] || `[${domino.high}|${domino.low}]`;
}

/**
 * Convert high/low values to Unicode glyph
 * @param high The high value of the domino
 * @param low The low value of the domino
 * @param vertical Whether to use vertical orientation (default: false)
 * @returns Unicode domino glyph or fallback text representation
 */
export function valuesToGlyph(high: number, low: number, vertical = false): string {
  const normalizedLow = Math.min(low, high);
  const normalizedHigh = Math.max(low, high);
  const key = `${normalizedLow}-${normalizedHigh}`;
  
  const glyphMap = vertical ? VERTICAL_DOMINO_GLYPHS : HORIZONTAL_DOMINO_GLYPHS;
  return glyphMap[key] || `[${high}|${low}]`;
}

/**
 * Check if the browser/environment supports domino glyphs
 * @returns true if domino glyphs are likely to render correctly
 */
export function supportsDominoGlyphs(): boolean {
  // Check if we're in a browser environment
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return false;
  }
  
  // Create a test canvas to check if the glyph renders
  try {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    if (!context) return false;
    
    // Test with a domino glyph
    const testGlyph = '🀰'; // 0-0 domino
    context.font = '16px serif';
    const metrics = context.measureText(testGlyph);
    
    // If the glyph doesn't render, it usually has very small or zero width
    return metrics.width > 8;
  } catch {
    return false;
  }
}