import { describe, test, expect } from 'vitest';
import { isValidPlay, getValidPlays } from '../../game/core/rules';
import type { Domino } from '../../game/types';

describe('Doubles Trump Follow Suit Rules', () => {
  const trump = 7; // Doubles are trump
  
  const currentTrick = [
    {
      player: 0,
      domino: { high: 6, low: 6, id: "6-6" } // P0 led double-six
    }
  ];
  
  const p1Hand: Domino[] = [
    { high: 4, low: 0, id: "4-0" },
    { high: 5, low: 5, id: "5-5" }, // Double-five (should be required to play)
    { high: 1, low: 0, id: "1-0" },
    { high: 6, low: 4, id: "6-4" },
    { high: 2, low: 1, id: "2-1" },
    { high: 4, low: 3, id: "4-3" },
    { high: 5, low: 0, id: "5-0" }
  ];

  test('Player must follow suit with doubles when doubles are trump', () => {
    // P1 has double-five and must play it when double-six was led
    const validPlays = getValidPlays(p1Hand, currentTrick, trump);
    
    expect(validPlays).toHaveLength(1);
    expect(validPlays[0].id).toBe("5-5");
  });

  test('Only doubles are valid when doubles trump is led', () => {
    const doubleFive = { high: 5, low: 5, id: "5-5" };
    const nonDouble = { high: 4, low: 0, id: "4-0" };
    
    expect(isValidPlay(doubleFive, p1Hand, currentTrick, trump)).toBe(true);
    expect(isValidPlay(nonDouble, p1Hand, currentTrick, trump)).toBe(false);
  });

  test('All doubles can follow when doubles trump is led', () => {
    const testHand: Domino[] = [
      { high: 0, low: 0, id: "0-0" },
      { high: 1, low: 1, id: "1-1" }, 
      { high: 2, low: 2, id: "2-2" },
      { high: 3, low: 3, id: "3-3" },
      { high: 4, low: 4, id: "4-4" },
      { high: 6, low: 1, id: "6-1" }, // Non-double
    ];
    
    const validPlays = getValidPlays(testHand, currentTrick, trump);
    
    // Should only return the doubles
    expect(validPlays).toHaveLength(5);
    expect(validPlays.every(d => d.high === d.low)).toBe(true);
  });

  test('When no doubles available, all plays are valid', () => {
    const handWithoutDoubles: Domino[] = [
      { high: 6, low: 1, id: "6-1" },
      { high: 4, low: 0, id: "4-0" },
      { high: 3, low: 2, id: "3-2" },
    ];
    
    const validPlays = getValidPlays(handWithoutDoubles, currentTrick, trump);
    
    // All plays should be valid when can't follow suit
    expect(validPlays).toHaveLength(3);
  });

  test('Bug report scenario - P1 should only be able to play 5-5', () => {
    // Exact scenario from bug report
    const bugReportHand: Domino[] = [
      { high: 4, low: 0, id: "4-0" },
      { high: 5, low: 5, id: "5-5" },
      { high: 1, low: 0, id: "1-0" },
      { high: 6, low: 4, id: "6-4" },
      { high: 2, low: 1, id: "2-1" },
      { high: 4, low: 3, id: "4-3" },
      { high: 5, low: 0, id: "5-0" }
    ];
    
    // Test each domino individually
    expect(isValidPlay({ high: 4, low: 0, id: "4-0" }, bugReportHand, currentTrick, trump)).toBe(false);
    expect(isValidPlay({ high: 5, low: 5, id: "5-5" }, bugReportHand, currentTrick, trump)).toBe(true);
    expect(isValidPlay({ high: 1, low: 0, id: "1-0" }, bugReportHand, currentTrick, trump)).toBe(false);
    expect(isValidPlay({ high: 6, low: 4, id: "6-4" }, bugReportHand, currentTrick, trump)).toBe(false);
    expect(isValidPlay({ high: 2, low: 1, id: "2-1" }, bugReportHand, currentTrick, trump)).toBe(false);
    expect(isValidPlay({ high: 4, low: 3, id: "4-3" }, bugReportHand, currentTrick, trump)).toBe(false);
    expect(isValidPlay({ high: 5, low: 0, id: "5-0" }, bugReportHand, currentTrick, trump)).toBe(false);
    
    // Get valid plays - should only be 5-5
    const validPlays = getValidPlays(bugReportHand, currentTrick, trump);
    expect(validPlays).toHaveLength(1);
    expect(validPlays[0].id).toBe("5-5");
  });
});