import { describe, it, expect } from 'vitest';
import { dominoToGlyph, valuesToGlyph, supportsDominoGlyphs } from '../../game/core/domino-glyphs';
import type { Domino } from '../../game/types';

describe('domino-glyphs', () => {
  describe('dominoToGlyph', () => {
    it('converts domino objects to Unicode glyphs', () => {
      const domino: Domino = { high: 6, low: 4, id: '6-4' };
      const glyph = dominoToGlyph(domino);
      expect(glyph).toBe('🁈'); // 4-6 horizontal glyph
    });

    it('handles reverse order (low-high)', () => {
      const domino: Domino = { high: 2, low: 5, id: '2-5' };
      const glyph = dominoToGlyph(domino);
      expect(glyph).toBe('🁀'); // 2-5 horizontal glyph
    });

    it('handles doubles correctly', () => {
      const domino: Domino = { high: 3, low: 3, id: '3-3' };
      const glyph = dominoToGlyph(domino);
      expect(glyph).toBe('🁂'); // 3-3 horizontal glyph
    });

    it('supports vertical orientation', () => {
      const domino: Domino = { high: 6, low: 6, id: '6-6' };
      const glyph = dominoToGlyph(domino, true);
      expect(glyph).toBe('🁧'); // 6-6 vertical glyph
    });

    it('falls back to text for invalid dominoes', () => {
      const domino: Domino = { high: 7, low: 8, id: '7-8' };
      const glyph = dominoToGlyph(domino);
      expect(glyph).toBe('[7|8]');
    });
  });

  describe('valuesToGlyph', () => {
    it('converts high/low values to glyphs', () => {
      expect(valuesToGlyph(0, 0)).toBe('🀰');
      expect(valuesToGlyph(1, 2)).toBe('🀸');
      expect(valuesToGlyph(5, 5)).toBe('🁉');
    });

    it('normalizes order automatically', () => {
      expect(valuesToGlyph(4, 2)).toBe('🀿'); // Same as 2-4
      expect(valuesToGlyph(2, 4)).toBe('🀿'); // Same as 2-4
    });
  });

  describe('supportsDominoGlyphs', () => {
    it('returns true for glyph support', () => {
      expect(supportsDominoGlyphs()).toBe(true);
    });
  });

  describe('all domino glyphs', () => {
    it('has all 28 dominoes represented', () => {
      const allDominoes: Array<[number, number]> = [];
      for (let i = 0; i <= 6; i++) {
        for (let j = i; j <= 6; j++) {
          allDominoes.push([i, j]);
        }
      }
      
      expect(allDominoes.length).toBe(28);
      
      // Check each domino has a glyph
      allDominoes.forEach(([low, high]) => {
        const glyph = valuesToGlyph(high, low);
        expect(glyph).not.toMatch(/^\[.*\]$/); // Should not be fallback format
        expect(glyph.length).toBe(2); // Unicode glyphs are 2 chars
      });
    });
  });
});