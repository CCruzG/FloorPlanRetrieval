import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";

const configDir = path.dirname(fileURLToPath(import.meta.url));

// Dev-only: the deployed server has nginx proxying these same paths to Flask
// (see project-71's buildweave.service for the pattern this mirrors). Locally,
// apps/web (Vite) and apps/api (Flask, default :5000) run as separate
// processes/ports, so relative fetches like /retrieve 404 without this.
const API_PROXY_TARGET = "http://127.0.0.1:5002";
const API_ROUTES = [
  "/config",
  "/retrieve",
  "/fit",
  "/fit_schema",
  "/arap_step",
  "/arap_canvas",
  "/feasibility_check",
  "/rerank_unit",
  "/fit_step1",
  "/render",
  "/comfy_health",
  "/generate_floor_plate",
  "/subdivide_floor_plate",
  "/swap_nodes",
  "/topo_check",
];

export default defineConfig({
  root: configDir,
  build: {
    outDir: path.resolve(configDir, "dist"),
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(
      API_ROUTES.map((route) => [route, API_PROXY_TARGET]),
    ),
  },
});
