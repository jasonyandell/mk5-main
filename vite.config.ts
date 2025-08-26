import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  base: process.env.NODE_ENV === 'production' ? '/mk5-main/' : '/',
  server: {
    port: parseInt(process.env.PORT || '60101'),
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})