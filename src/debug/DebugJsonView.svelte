<script lang="ts">
  import type { GameState } from '../game/types';
  
  export let state: GameState;
  
  let prettyJson: string;
  let copied = false;
  
  $: prettyJson = JSON.stringify(state, null, 2);
  
  function copyToClipboard() {
    navigator.clipboard.writeText(prettyJson).then(() => {
      copied = true;
      setTimeout(() => copied = false, 2000);
    });
  }
  
  function downloadJson() {
    const blob = new Blob([prettyJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `game-state-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  function highlightJson(json: string): string {
    return json
      .replace(/("[\w\d_-]+")\s*:/g, '<span class="json-key">$1</span>:')
      .replace(/:\s*(".*?")/g, ': <span class="json-string">$1</span>')
      .replace(/:\s*(\d+)/g, ': <span class="json-number">$1</span>')
      .replace(/:\s*(true|false)/g, ': <span class="json-boolean">$1</span>')
      .replace(/:\s*(null)/g, ': <span class="json-null">$1</span>');
  }
</script>

<div class="json-view-panel">
  <div class="json-header">
    <h3>Raw State JSON</h3>
    <div class="json-controls">
      <button 
        class="json-btn"
        on:click={copyToClipboard}
        data-testid="copy-json"
      >
        {copied ? '✓ Copied!' : '📋 Copy'}
      </button>
      <button 
        class="json-btn"
        on:click={downloadJson}
        data-testid="download-json"
      >
        💾 Download
      </button>
    </div>
  </div>
  
  <div class="json-content">
    <pre class="json-display">{@html highlightJson(prettyJson)}</pre>
  </div>
  
  <div class="json-info">
    <div class="info-item">
      <label>Size</label>
      <span>{(new Blob([prettyJson]).size / 1024).toFixed(2)} KB</span>
    </div>
    <div class="info-item">
      <label>Keys</label>
      <span>{Object.keys(state).length}</span>
    </div>
    <div class="info-item">
      <label>Players</label>
      <span>{state.players.length}</span>
    </div>
    <div class="info-item">
      <label>Tricks</label>
      <span>{state.tricks.length}</span>
    </div>
  </div>
</div>

<style>
  .json-view-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  
  .json-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  
  .json-header h3 {
    margin: 0;
    color: #4CAF50;
    font-size: 18px;
  }
  
  .json-controls {
    display: flex;
    gap: 10px;
  }
  
  .json-btn {
    padding: 8px 16px;
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    color: #fff;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  
  .json-btn:hover {
    background: #444;
    border-color: #666;
  }
  
  .json-content {
    flex: 1;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 20px;
    overflow: auto;
    margin-bottom: 20px;
  }
  
  .json-display {
    margin: 0;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.5;
    color: #e0e0e0;
    white-space: pre;
  }
  
  :global(.json-key) {
    color: #9CDCFE;
  }
  
  :global(.json-string) {
    color: #CE9178;
  }
  
  :global(.json-number) {
    color: #B5CEA8;
  }
  
  :global(.json-boolean) {
    color: #569CD6;
  }
  
  :global(.json-null) {
    color: #569CD6;
  }
  
  .json-info {
    display: flex;
    gap: 30px;
    padding: 15px;
    background: #333;
    border-radius: 6px;
  }
  
  .info-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  
  .info-item label {
    font-size: 12px;
    color: #888;
  }
  
  .info-item span {
    font-size: 16px;
    color: #fff;
    font-weight: 500;
  }
  
  ::-webkit-scrollbar {
    width: 12px;
    height: 12px;
  }
  
  ::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 6px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 6px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: #666;
  }
  
  ::-webkit-scrollbar-corner {
    background: #1a1a1a;
  }
</style>