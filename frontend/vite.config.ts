import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  // Read VITE_* variables from the single root .env (only VITE_-prefixed ones reach the browser)
  envDir: '..',
  plugins: [react(), tailwindcss()],
})
