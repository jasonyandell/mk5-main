export class GameTestHelper {
  static createMockGameState() {
    return {
      players: [
        { id: 0, hand: [], partner: 2 },
        { id: 1, hand: [], partner: 3 },
        { id: 2, hand: [], partner: 0 },
        { id: 3, hand: [], partner: 1 },
      ],
      currentBid: null,
      currentTrick: [],
      trickWinner: null,
      scores: { team1: 0, team2: 0 },
      marks: { team1: 0, team2: 0 },
    };
  }

  static createDomino(end1: number, end2: number) {
    return { end1, end2 };
  }

  static createCountingDominoes() {
    return [
      { end1: 5, end2: 5 }, // 10 points
      { end1: 6, end2: 4 }, // 10 points
      { end1: 5, end2: 0 }, // 5 points
      { end1: 4, end2: 1 }, // 5 points
      { end1: 3, end2: 2 }, // 5 points
    ];
  }

  static dealDominoes() {
    // Create standard double-six set
    const dominoes = [];
    for (let i = 0; i <= 6; i++) {
      for (let j = i; j <= 6; j++) {
        dominoes.push({ end1: i, end2: j });
      }
    }
    return dominoes;
  }
}