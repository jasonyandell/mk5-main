import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 60101,
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})