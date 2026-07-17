import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Static SPA with hash routing — `base: "./"` makes the build servable from any
// subpath (GitHub Pages, Netlify preview) with zero server configuration.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          echarts: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers"],
        },
      },
    },
  },
});
