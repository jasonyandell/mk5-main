<script lang="ts">
  import { gameActions } from '../ui/stores/gameStore';
  import type { GameState } from '../game/types';
  
  let uploadInput: HTMLInputElement;
  
  function loadFromFile() {
    uploadInput?.click();
  }
  
  function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const state: GameState = JSON.parse(content);
        gameActions.loadState(state);
      } catch (error) {
        alert('Invalid state file format');
      }
    };
    reader.readAsText(file);
  }
  
  function generateTestScenarios() {
    const scenarios = [
      {
        name: 'Bidding Phase - Opening Round',
        description: 'Fresh deal with all players able to bid',
        setup: () => gameActions.resetGame()
      },
      {
        name: 'High Bidding Scenario',
        description: 'Simulate high-stakes bidding with marks',
        setup: () => {
          // Would need to create specific game states
          alert('Test scenario generation coming soon!');
        }
      },
      {
        name: 'Trump Selection',
        description: 'Player has won bid and needs to select trump',
        setup: () => {
          alert('Test scenario generation coming soon!');
        }
      },
      {
        name: 'Mid-Game Trick',
        description: 'Players are in middle of a trick',
        setup: () => {
          alert('Test scenario generation coming soon!');
        }
      },
      {
        name: 'End Game Scenario',
        description: 'Teams are close to winning',
        setup: () => {
          alert('Test scenario generation coming soon!');
        }
      }
    ];
    
    return scenarios;
  }
  
  const testScenarios = $derived(generateTestScenarios());
</script>

<div class="test-generator">
  <h3>Test State Generator</h3>
  
  <div class="file-operations">
    <h4>State File Operations</h4>
    <div class="file-buttons">
      <button onclick={loadFromFile}>Load State from File</button>
      <input 
        bind:this={uploadInput}
        type="file" 
        accept=".json"
        style="display: none"
        onchange={handleFileUpload}
      />
    </div>
  </div>
  
  <div class="test-scenarios">
    <h4>Test Scenarios</h4>
    <div class="scenarios-list">
      {#each testScenarios as scenario}
        <div class="scenario">
          <div class="scenario-info">
            <h5>{scenario.name}</h5>
            <p>{scenario.description}</p>
          </div>
          <button onclick={scenario.setup}>Load</button>
        </div>
      {/each}
    </div>
  </div>
  
  <div class="url-injection">
    <h4>URL State Injection</h4>
    <p class="url-info">
      You can inject game states via URL parameters for testing specific scenarios.
      Add <code>?state=base64encodedstate</code> to the URL.
    </p>
    <button onclick={() => alert('URL injection documentation coming soon!')}>
      Learn More
    </button>
  </div>
</div>

<style>
  .test-generator {
    border-top: 1px solid #eee;
    padding-top: 20px;
  }
  
  .test-generator h3 {
    margin: 0 0 20px 0;
    color: #333;
  }
  
  .test-generator h4 {
    margin: 16px 0 8px 0;
    color: #495057;
    font-size: 14px;
  }
  
  .file-operations {
    margin-bottom: 20px;
  }
  
  .file-buttons {
    display: flex;
    gap: 8px;
  }
  
  .file-buttons button {
    padding: 8px 16px;
    border: 1px solid #007bff;
    border-radius: 4px;
    background: #007bff;
    color: white;
    cursor: pointer;
    font-size: 12px;
  }
  
  .file-buttons button:hover {
    background: #0056b3;
  }
  
  .scenarios-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .scenario {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    background: #f8f9fa;
  }
  
  .scenario-info {
    flex: 1;
  }
  
  .scenario-info h5 {
    margin: 0 0 4px 0;
    color: #212529;
    font-size: 13px;
  }
  
  .scenario-info p {
    margin: 0;
    color: #6c757d;
    font-size: 11px;
  }
  
  .scenario button {
    padding: 6px 12px;
    border: 1px solid #28a745;
    border-radius: 4px;
    background: #28a745;
    color: white;
    cursor: pointer;
    font-size: 11px;
  }
  
  .scenario button:hover {
    background: #1e7e34;
  }
  
  .url-injection {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 6px;
    padding: 12px;
    margin-top: 16px;
  }
  
  .url-info {
    margin: 0 0 8px 0;
    font-size: 12px;
    color: #856404;
  }
  
  .url-info code {
    background: rgba(0, 0, 0, 0.1);
    padding: 2px 4px;
    border-radius: 2px;
    font-family: monospace;
  }
  
  .url-injection button {
    padding: 6px 12px;
    border: 1px solid #856404;
    border-radius: 4px;
    background: transparent;
    color: #856404;
    cursor: pointer;
    font-size: 11px;
  }
  
  .url-injection button:hover {
    background: rgba(133, 100, 4, 0.1);
  }
</style>