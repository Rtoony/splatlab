import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5174,
    // dev convenience: proxy API + supersplat to the splatlab backend on :3416
    proxy: {
      "/api": "http://127.0.0.1:3416",
      "/supersplat": "http://127.0.0.1:3416",
    },
  },
});
