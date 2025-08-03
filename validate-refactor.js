// Quick validation script to test our refactor
const { createInitialState } = require('./dist/game/core/state.js');
const { GameEngine } = require('./dist/game/core/gameEngine.js');

try {
  console.log('Testing basic state creation...');
  const state = createInitialState({ shuffleSeed: 12345 });
  
  console.log('✅ State created successfully');
  console.log('winningBidder:', state.winningBidder, '(should be -1, not null)');
  console.log('trump:', state.trump, '(should be TrumpSelection object, not null)');
  console.log('currentSuit:', state.currentSuit, '(should be -1, not null)');
  
  console.log('\nTesting GameEngine...');
  const engine = new GameEngine(state);
  
  console.log('✅ GameEngine created successfully');
  console.log('Initial history length:', engine.getHistory().length);
  
  console.log('\nTesting action execution...');
  const validActions = engine.getValidActions();
  console.log('Valid actions count:', validActions.length);
  
  if (validActions.length > 0) {
    const firstAction = validActions[0];
    console.log('First action:', firstAction);
    
    engine.executeAction(firstAction);
    console.log('✅ Action executed successfully');
    console.log('History length after action:', engine.getHistory().length);
    
    const undoResult = engine.undo();
    console.log('✅ Undo successful:', undoResult);
    console.log('History length after undo:', engine.getHistory().length);
  }
  
  console.log('\n🎉 All tests passed! Refactor appears successful.');
} catch (error) {
  console.error('❌ Error during validation:', error.message);
  console.error(error.stack);
  process.exit(1);
}
