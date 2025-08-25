import { vitePreprocess } from '@sveltejs/vite-plugin-svelte'

export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    warningFilter: (warning) => {
      // Suppress CSS unused selector warnings for animation classes
      if (warning.code === 'css_unused_selector' && 
          warning.message.includes('animate-')) {
        return false;
      }
      // Suppress CSS unused selector warnings for utility classes
      if (warning.code === 'css_unused_selector' && 
          (warning.message.includes('tap-') || 
           warning.message.includes('touch-manipulation') ||
           warning.message.includes('drawer-transition'))) {
        return false;
      }
      // Suppress unknown at-rule warnings for Tailwind's @apply
      if (warning.code === 'css-unknown-at-rule-name' && 
          warning.message.includes('@apply')) {
        return false;
      }
      // Also suppress "Unknown at rule" warnings for @apply
      if (warning.message && warning.message.includes('Unknown at rule @apply')) {
        return false;
      }
      return true;
    }
  }
}