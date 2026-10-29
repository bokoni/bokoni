import { defineConfig } from "vite";

export default defineConfig({
  // Relative paths: the built site works from any directory (GitHub Pages…).
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 4096 },
});
