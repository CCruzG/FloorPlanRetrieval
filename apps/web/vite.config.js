import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const configDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: configDir,
  build: {
    outDir: path.resolve(configDir, 'dist'),
    emptyOutDir: true,
  },
});
