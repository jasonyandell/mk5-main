<script lang="ts">
  import Icon from '../icons/Icon.svelte';
  
  export let data: any;
  export let path: string = 'root';
  export let searchQuery: string = '';
  export let changedPaths: Set<string> = new Set();
  
  let expanded: { [key: string]: boolean } = {};
  
  // Check if this path or any child path has changes
  function hasChanges(currentPath: string): boolean {
    if (changedPaths.has(currentPath)) return true;
    
    // Check if any child path has changes
    for (const path of changedPaths) {
      if (path.startsWith(currentPath + '.')) {
        return true;
      }
    }
    return false;
  }
  
  // Check if value matches search query
  function matchesSearch(key: string, value: any): boolean {
    if (!searchQuery) return true;
    
    const lowerQuery = searchQuery.toLowerCase();
    if (key.toLowerCase().includes(lowerQuery)) return true;
    
    if (typeof value === 'string' && value.toLowerCase().includes(lowerQuery)) return true;
    if (typeof value === 'number' && value.toString().includes(searchQuery)) return true;
    
    return false;
  }
  
  // Filter object based on search
  function filterData(obj: any): any {
    if (!searchQuery) return obj;
    
    if (Array.isArray(obj)) {
      return obj.filter((_, index) => matchesSearch(index.toString(), obj[index]));
    }
    
    const filtered: any = {};
    for (const [key, value] of Object.entries(obj)) {
      if (matchesSearch(key, value)) {
        filtered[key] = value;
      } else if (typeof value === 'object' && value !== null) {
        const subFiltered = filterData(value);
        if (Object.keys(subFiltered).length > 0 || Array.isArray(subFiltered) && subFiltered.length > 0) {
          filtered[key] = value;
        }
      }
    }
    return filtered;
  }
  
  // Toggle expansion
  function toggle(key: string) {
    expanded[key] = !expanded[key];
  }
  
  // Copy value to clipboard
  function copyValue(value: any) {
    navigator.clipboard.writeText(JSON.stringify(value, null, 2));
  }
  
  $: filteredData = filterData(data);
  $: dataEntries = Object.entries(filteredData);
</script>

{#if typeof data === 'object' && data !== null}
  {#if Array.isArray(data)}
    <div class="array-container">
      {#each filteredData as item, index}
        {@const itemPath = `${path}[${index}]`}
        {@const isObject = typeof item === 'object' && item !== null}
        {@const isExpanded = expanded[itemPath]}
        {@const hasChange = hasChanges(itemPath)}
        
        <div class="tree-item" class:changed={hasChange}>
          <div class="tree-key">
            {#if isObject}
              <button 
                class="expand-button"
                on:click={() => toggle(itemPath)}
              >
                {isExpanded ? '▼' : '▶'}
              </button>
            {/if}
            <span class="key-name">[{index}]</span>
            {#if !isObject}
              <span class="value-preview">{JSON.stringify(item)}</span>
            {:else}
              <span class="type-label">{Array.isArray(item) ? 'array' : 'object'}</span>
            {/if}
            <button 
              class="copy-button"
              on:click={() => copyValue(item)}
              title="Copy value"
            >
              <Icon name="clipboard" size="sm" />
            </button>
          </div>
          
          {#if isObject && isExpanded}
            <div class="tree-children">
              <svelte:self data={item} path={itemPath} {searchQuery} {changedPaths} />
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {:else}
    <div class="object-container">
      {#each dataEntries as [key, value]}
        {@const itemPath = `${path}.${key}`}
        {@const isObject = typeof value === 'object' && value !== null}
        {@const isExpanded = expanded[itemPath]}
        {@const hasChange = hasChanges(itemPath)}
        
        <div class="tree-item" class:changed={hasChange}>
          <div class="tree-key">
            {#if isObject}
              <button 
                class="expand-button"
                on:click={() => toggle(itemPath)}
              >
                {isExpanded ? '▼' : '▶'}
              </button>
            {/if}
            <span class="key-name">{key}:</span>
            {#if !isObject}
              <span class="value-preview">{JSON.stringify(value)}</span>
            {:else}
              <span class="type-label">{Array.isArray(value) ? `array[${value.length}]` : 'object'}</span>
            {/if}
            <button 
              class="copy-button"
              on:click={() => copyValue(value)}
              title="Copy value"
            >
              <Icon name="clipboard" size="sm" />
            </button>
          </div>
          
          {#if isObject && isExpanded}
            <div class="tree-children">
              <svelte:self data={value} path={itemPath} {searchQuery} {changedPaths} />
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
{:else}
  <span class="primitive-value">{JSON.stringify(data)}</span>
{/if}

<style>
  .tree-item {
    margin: 2px 0;
  }
  
  .tree-key {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px;
    border-radius: 4px;
    transition: background-color 0.2s;
  }
  
  .tree-key:hover {
    background-color: #f3f4f6;
  }
  
  .tree-item.changed > .tree-key {
    background-color: #fef3c7;
    animation: highlight 1s ease-out;
  }
  
  @keyframes highlight {
    from { background-color: #fde047; }
    to { background-color: #fef3c7; }
  }
  
  .expand-button {
    width: 20px;
    height: 20px;
    padding: 0;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 12px;
    color: #6b7280;
  }
  
  .expand-button:hover {
    color: #374151;
  }
  
  .key-name {
    font-weight: 600;
    color: #4b5563;
    font-family: monospace;
    font-size: 13px;
  }
  
  .value-preview {
    color: #059669;
    font-family: monospace;
    font-size: 13px;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .type-label {
    color: #6b7280;
    font-size: 12px;
    font-style: italic;
  }
  
  .copy-button {
    margin-left: auto;
    padding: 2px 6px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
  }
  
  .tree-key:hover .copy-button {
    opacity: 1;
  }
  
  .copy-button:hover {
    background-color: #e5e7eb;
    border-radius: 4px;
  }
  
  .tree-children {
    margin-left: 24px;
    border-left: 1px solid #e5e7eb;
    padding-left: 12px;
  }
  
  .primitive-value {
    color: #059669;
    font-family: monospace;
    font-size: 13px;
  }
  
  .array-container,
  .object-container {
    margin: 0;
  }
</style>