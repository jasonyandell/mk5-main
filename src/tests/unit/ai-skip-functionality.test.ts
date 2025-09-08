import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AIController } from '../../game/controllers/AIController';
import { ControllerManager } from '../../game/controllers/ControllerManager';
import { createInitialState } from '../../game/core/state';
import type { StateTransition } from '../../game/types';

describe('AI Skip Functionality', () => {
  let aiController: AIController;
  let controllerManager: ControllerManager;
  let mockExecuteTransition: ReturnType<typeof vi.fn>;
  
  beforeEach(() => {
    mockExecuteTransition = vi.fn();
    aiController = new AIController(1, mockExecuteTransition);
    controllerManager = new ControllerManager(mockExecuteTransition);
  });

  it('should store pending decision when AI starts thinking', () => {
    const state = createInitialState();
    const mockTransition: StateTransition = {
      id: 'bid-30',
      label: 'Bid 30',
      action: { type: 'bid', player: 1, bid: 'points', value: 30 },
      newState: state
    };

    const transitions = [mockTransition];
    
    aiController.onStateChange(state, transitions);
    
    // AI should have stored the pending decision
    expect(aiController['pendingDecision']).toBeDefined();
    expect(aiController['pendingDecision']?.transition).toBe(mockTransition);
  });

  it('should execute immediately when executeNow is called', () => {
    const state = createInitialState();
    const mockTransition: StateTransition = {
      id: 'bid-30',
      label: 'Bid 30',
      action: { type: 'bid', player: 1, bid: 'points', value: 30 },
      newState: state
    };

    const transitions = [mockTransition];
    
    aiController.onStateChange(state, transitions);
    
    // Verify AI has pending decision
    expect(aiController['pendingDecision']).toBeDefined();
    expect(aiController['pendingDecision']?.transition).toBe(mockTransition);
    
    // Execute immediately
    aiController.executeNow(state);
    
    // Verify the transition was executed
    expect(mockExecuteTransition).toHaveBeenCalledWith(mockTransition);
    expect(aiController['pendingDecision']).toBeUndefined();
  });

  it('should not execute if no pending transition', () => {
    const state = createInitialState();
    aiController.executeNow(state);
    expect(mockExecuteTransition).not.toHaveBeenCalled();
  });

  it('should skip delays for all AI controllers', () => {
    // Set up multiple AI controllers
    controllerManager.setupLocalGame([
      { type: 'human' },
      { type: 'ai' },
      { type: 'ai' },
      { type: 'ai' }
    ]);

    const state = createInitialState();
    
    // No timers needed anymore
    
    // Trigger state change to make AIs think
    controllerManager.onStateChange(state);
        
    // Should work without throwing
    // The actual execution happens through the controllers
  });

  it('should work without timers', () => {
    const state = createInitialState();
    const mockTransition: StateTransition = {
      id: 'bid-30',
      label: 'Bid 30',
      action: { type: 'bid', player: 1, bid: 'points', value: 30 },
      newState: state
    };

    const transitions = [mockTransition];
    
    aiController.onStateChange(state, transitions);
    
    // Verify AI has pending decision
    expect(aiController['pendingDecision']).toBeDefined();
    
    // Destroy the controller
    aiController.destroy();
    
    // No cleanup needed anymore since no timers
    // The test passes if destroy doesn't throw
  });
});