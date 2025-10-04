import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  console.log('Building in mode:', mode);
  
  return {
    plugins: [react()],
    server: {
      port: 3000,
      host: true,
      // Enable CORS for development
      cors: true
    },
    build: {
      outDir: 'dist',
      sourcemap: mode === 'development',
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom'],
            charts: ['recharts']
          }
        }
      },
      // Optimize for production
      minify: mode === 'production' ? 'terser' : false,
      target: 'es2015'
    },
    // Define environment variables for the build process
    define: {
      // Expose environment info to the app
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
      __BUILD_DATE__: JSON.stringify(new Date().toISOString())
    },
    // Preview server configuration (for vercel preview)
    preview: {
      port: 3000,
      host: true
    }
  }
})
