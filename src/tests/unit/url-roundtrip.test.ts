/**
 * URL Roundtrip Tests - Verify state ↔ URL isomorphism
 *
 * These tests verify that:
 * 1. Any game state can be serialized to a URL
 * 2. That URL can be used to recreate identical state
 * 3. initialHands and seed are properly handled as mutually exclusive
 */

import { describe, it, expect } from 'vitest';
import { stateToUrl, decodeGameUrl, encodeGameUrl } from '../../game/core/url-compression';
import { replayFromUrl } from '../../game/utils/urlReplay';
import { HeadlessRoom } from '../../server/HeadlessRoom';
import type { GameConfig } from '../../game/types/config';
import { generateDealFromConstraints } from '../helpers/dealConstraints';

// Deterministic hands for URL encoding tests
// Using dealConstraints with fixed seed ensures same hands every run
const FIXED_HANDS = generateDealFromConstraints({ fillSeed: 42 }).map(hand =>
  hand.map(d => ({ high: d.high, low: d.low, id: d.id, points: 0 }))
);

describe('URL Roundtrip Isomorphism', () => {
  describe('seed-based games', () => {
    it('should roundtrip a fresh game with seed', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        shuffleSeed: 12345
      };
      const room = new HeadlessRoom(config, 12345);
      const state = room.getState();

      // State → URL
      const url = stateToUrl(state);

      // URL → State
      const result = replayFromUrl(url);

      // Verify identical state
      expect(result.state.shuffleSeed).toBe(state.shuffleSeed);
      expect(result.state.phase).toBe(state.phase);
      expect(result.state.dealer).toBe(state.dealer);
      expect(result.state.players.map(p => p.hand)).toEqual(state.players.map(p => p.hand));
    });

    it('should roundtrip a game with actions', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        shuffleSeed: 12345
      };
      const room = new HeadlessRoom(config, 12345);

      // Play some actions
      room.executeAction(0, { type: 'bid', bid: 'points', value: 30, player: 0 });
      room.executeAction(1, { type: 'pass', player: 1 });
      room.executeAction(2, { type: 'pass', player: 2 });
      room.executeAction(3, { type: 'pass', player: 3 });

      const state = room.getState();

      // State → URL
      const url = stateToUrl(state);

      // Verify URL contains actions
      expect(url).toContain('a=');

      // URL → State
      const result = replayFromUrl(url);

      // Verify identical state
      expect(result.state.phase).toBe(state.phase);
      expect(result.state.currentBid).toEqual(state.currentBid);
      expect(result.state.winningBidder).toBe(state.winningBidder);
      expect(result.state.actionHistory.length).toBe(state.actionHistory.length);
    });
  });

  describe('initialHands-based games', () => {
    it('should roundtrip a game with explicit hands', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        dealOverrides: { initialHands: FIXED_HANDS }
      };
      const room = new HeadlessRoom(config);
      const state = room.getState();

      // State → URL
      const url = stateToUrl(state);

      // Verify URL uses 'i=' not 's='
      expect(url).toContain('i=');
      expect(url).not.toContain('s=');

      // URL → State
      const result = replayFromUrl(url);

      // Compare hands by high/low/id (ignore points field which is computed)
      const extractCore = (hand: typeof state.players[0]['hand']) =>
        hand.map(d => ({ high: d.high, low: d.low, id: d.id }));

      expect(result.state.players.map(p => extractCore(p.hand)))
        .toEqual(state.players.map(p => extractCore(p.hand)));
    });

    it('should encode initialHands compactly (~24 chars)', () => {
      const url = encodeGameUrl(undefined, [], ['human', 'ai', 'ai', 'ai'], 3, undefined, undefined, undefined, undefined, FIXED_HANDS);

      // Extract just the 'i=' parameter value
      const params = new URLSearchParams(url);
      const iParam = params.get('i');

      expect(iParam).toBeDefined();
      expect(iParam!.length).toBeLessThanOrEqual(24);
    });
  });

  describe('mutual exclusion: seed vs initialHands', () => {
    it('encodeGameUrl with initialHands should omit seed', () => {
      // Even if seed is provided, initialHands takes precedence
      const url = encodeGameUrl(12345, [], undefined, undefined, undefined, undefined, undefined, undefined, FIXED_HANDS);

      expect(url).toContain('i=');
      expect(url).not.toContain('s=');
    });

    it('decodeGameUrl should return initialHands when i= present', () => {
      const url = encodeGameUrl(undefined, [], undefined, undefined, undefined, undefined, undefined, undefined, FIXED_HANDS);
      const decoded = decodeGameUrl(url);

      expect(decoded.initialHands).toBeDefined();
      expect(decoded.initialHands!.length).toBe(4);
      expect(decoded.seed).toBe(0); // seed should be 0 when initialHands present
    });

    it('decodeGameUrl should return seed when s= present', () => {
      const url = encodeGameUrl(12345, []);
      const decoded = decodeGameUrl(url);

      expect(decoded.seed).toBe(12345);
      expect(decoded.initialHands).toBeUndefined();
    });
  });

  describe('layers encoding', () => {
    it('should encode/decode layers via encodeGameUrl/decodeGameUrl directly', () => {
      // Layers roundtrip works at the URL level
      const url = encodeGameUrl(12345, [], ['human', 'ai', 'ai', 'ai'], 3, undefined, undefined, undefined, ['nello', 'plunge']);

      expect(url).toContain('l=');

      const decoded = decodeGameUrl(url);
      expect(decoded.layers).toEqual(['nello', 'plunge']);
    });

    it('should roundtrip layers through stateToUrl', () => {
      // Create game WITH layers - they should now be persisted to initialConfig
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        shuffleSeed: 12345,
        layers: ['nello', 'plunge']
      };
      const room = new HeadlessRoom(config, 12345);
      const state = room.getState();

      // Verify layers are in initialConfig
      expect(state.initialConfig.layers).toEqual(['nello', 'plunge']);

      // State → URL should include layers
      const url = stateToUrl(state);
      expect(url).toContain('l=');

      // URL → State should restore layers in config
      const result = replayFromUrl(url);
      expect(result.config.layers).toEqual(['nello', 'plunge']);
    });

    it('should not include empty layers array in URL', () => {
      // Game without layers
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        shuffleSeed: 12345
      };
      const room = new HeadlessRoom(config, 12345);
      const state = room.getState();

      // URL should NOT have l= param when no layers
      const url = stateToUrl(state);
      expect(url).not.toContain('l=');
    });
  });
});
