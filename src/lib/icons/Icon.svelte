<script lang="ts">
  import { icons } from './icons';
  
  export let name: keyof typeof icons;
  export let size: 'sm' | 'md' | 'lg' | 'xl' = 'md';
  export let className: string = '';
  export let ariaLabel: string = '';
  
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
    xl: 'w-8 h-8'
  };
  
  $: svgContent = icons[name] || '';
</script>

<span 
  class="{sizeClasses[size]} {className} inline-block"
  aria-label={ariaLabel}
  role={ariaLabel ? 'img' : undefined}
>
  {@html svgContent}
</span>

<style>
  span :global(svg) {
    width: 100%;
    height: 100%;
  }
</style>