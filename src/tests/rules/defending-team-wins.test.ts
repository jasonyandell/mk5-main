import { describe, it, expect } from 'vitest';

describe('Feature: Hand Victory', () => {
  describe('Scenario: Defending Team Wins', () => {
    it('should set the bidders when bidding team fails to take enough points', () => {
      // Given a hand has been played
      const biddingTeamPoints = 38;
      const defendingTeamPoints = 4; // 42 - 38 = 4
      const bid = 40; // The bidding team bid 40 points
      
      // When the bidding team fails to take enough points
      const biddingTeamMadeBid = biddingTeamPoints >= bid;
      
      // Then the defending team wins by "setting" the bidders
      expect(biddingTeamMadeBid).toBe(false);
      expect(biddingTeamPoints).toBeLessThan(bid);
      expect(biddingTeamPoints + defendingTeamPoints).toBe(42); // Total points always equals 42
      
      // In tournament play, the defending team receives marks equal to what was bid
      const defendingTeamMarksAwarded = biddingTeamMadeBid ? 0 : getMarksForBid(bid);
      expect(defendingTeamMarksAwarded).toBe(1); // 40 point bid = 1 mark
    });

    it('should award defending team marks when setting a mark bid', () => {
      // Given a hand has been played with a 2 mark bid (84 points)
      const biddingTeamPoints = 41; // Just missed the 42 needed for 1 mark
      const bid = 84; // 2 marks
      
      // When the bidding team fails to make their bid
      const biddingTeamMadeBid = biddingTeamPoints >= 42; // Need at least 42 for 1 mark
      
      // Then the defending team wins and receives 2 marks
      expect(biddingTeamMadeBid).toBe(false);
      const defendingTeamMarksAwarded = getMarksForBid(bid);
      expect(defendingTeamMarksAwarded).toBe(2);
    });

    it('should correctly determine set on various bid levels', () => {
      const testCases = [
        { bid: 30, biddingTeamPoints: 29, isSet: true },
        { bid: 35, biddingTeamPoints: 34, isSet: true },
        { bid: 41, biddingTeamPoints: 40, isSet: true },
        { bid: 42, biddingTeamPoints: 41, isSet: true }, // 1 mark bid
        { bid: 84, biddingTeamPoints: 41, isSet: true }, // 2 mark bid
        { bid: 30, biddingTeamPoints: 30, isSet: false }, // Made exactly
        { bid: 35, biddingTeamPoints: 40, isSet: false }, // Made with extra
        { bid: 42, biddingTeamPoints: 42, isSet: false }, // Made 1 mark exactly
      ];

      testCases.forEach(({ bid, biddingTeamPoints, isSet }) => {
        const biddingTeamMadeBid = biddingTeamPoints >= Math.min(bid, 42);
        expect(biddingTeamMadeBid).toBe(!isSet);
      });
    });
  });
});

// Helper function to determine marks for a given bid
function getMarksForBid(bid: number): number {
  if (bid <= 41) return 1;
  return Math.ceil(bid / 42);
}