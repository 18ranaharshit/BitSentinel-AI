import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3005,
    proxy: {
      '/api': 'http://localhost:8005',
      '/ws': {
        target: 'ws://localhost:8005',
        ws: true
      }
    }
  }
})
