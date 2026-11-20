import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";

import preact from "@preact/preset-vite";
import { type Connect, defineConfig, type Plugin } from "vite";

const MODELS_DIR = join(__dirname, "models");
const TYPES: Record<string, string> = { ".json": "application/json", ".jinja": "text/plain" };

/**
 * Serve web/models/ (the exported model, see scripts/export_onnx.py) at /models/ in `vite dev` and
 * `vite preview`. It is kept out of public/ so that builds do not copy 1.4 GB into dist/; the
 * deployed site loads the model from the Hugging Face Hub instead.
 */
function serveModels(): Plugin {
  const middleware: Connect.NextHandleFunction = (req, res, next) => {
    const path = decodeURIComponent((req.url ?? "").split("?")[0]!);
    const file = normalize(join(MODELS_DIR, path));
    if (!file.startsWith(MODELS_DIR + sep)) return next();
    let size: number;
    try {
      const stat = statSync(file);
      if (!stat.isFile()) return next();
      size = stat.size;
    } catch {
      return next();
    }
    res.setHeader("Content-Type", TYPES[extname(file)] ?? "application/octet-stream");
    res.setHeader("Content-Length", String(size));
    createReadStream(file).pipe(res);
  };
  return {
    name: "serve-models",
    configureServer: (server) => void server.middlewares.use("/models", middleware),
    configurePreviewServer: (server) => void server.middlewares.use("/models", middleware),
  };
}

export default defineConfig({
  base: "./",
  plugins: [preact(), serveModels()],
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
