# conlang-gym website

A static site where a person, or the trained model, plays conlang-gym episodes in the browser. The model runs on the visitor's GPU with WebGPU; there is no server.

| Path | Contents |
|---|---|
| `src/env/` | TypeScript port of `conlang_gym/language.py` and `solver.py`. A seed gives the same language, prompts, replies, scores and solver result as the Python code |
| `src/env/dataset_seeds.json` | Which seeds were validation or training data of the published model |
| `src/llm/engine.ts` | Greedy decoding of the ONNX model with onnxruntime-web (WebGPU), keeping the KV cache and Gated DeltaNet states between turns |
| `src/llm/agent.ts` | The agent loop of `conlang_gym/agent_loop.py` (non-thinking mode): the same token stream and token limits as in training |
| `src/llm/worker.ts` | Web Worker that downloads the model (kept with the Cache API) and runs it |
| `src/ui/` | The page (Preact) |
| `scripts/export_onnx.py` | Converts the trained checkpoint to the ONNX files the page loads |
| `tests/` | Parity tests against fixtures dumped from the Python code, and tests of the agent loop |

## Development

```bash
cd web
npm ci
uv run scripts/export_onnx.py --out_dir models/conlang-gym-2b   # once: writes the model (1.4 GB)
npm run dev
```

`npm run dev` and `npm run preview` serve `web/models/` at `/models/`, and in development the page loads the model from there. The model needs WebGPU with the `shader-f16` feature (current Chrome or Edge on desktop). On an Apple M4 it decodes about 10 tokens/s, and an episode takes 30–60 s.

```bash
npm test               # parity with the Python environment, and the agent loop
npx tsc --noEmit
```

The fixtures in `tests/fixtures/` and `src/env/dataset_seeds.json` are generated from the Python code (Python 3.12, from the repository root):

```bash
PYTHONPATH=. python web/scripts/dump_env_fixtures.py
PYTHONPATH=. python web/scripts/dataset_seeds.py
```

## The model files

`scripts/export_onnx.py` takes the graph of [onnx-community/Qwen3.5-2B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT): `embed_tokens_q4f16` and `decoder_model_merged_q4f16`, without the vision encoder. It replaces every weight with the trained checkpoint's, quantized the same way (4-bit MatMulNBits, blocks of 32, with zero points). `--check` rebuilds the reference files from Qwen/Qwen3.5-2B and checks that all 697 weights come out bit-identical, which verifies the conversion.

```bash
uv run scripts/export_onnx.py --check
```

The output (`onnx/`, the tokenizer files and `manifest.json` with each file's size and a content version) is 1.40 GB. `scripts/node_generate.ts` runs the same engine on CPU with onnxruntime-node, which is useful for comparing with Python:

```bash
npx vite-node scripts/node_generate.ts -- models/conlang-gym-2b prompt.txt 64
```

## Deployment

The site is static (`npm run build` writes `dist/`), but the model is too large to ship with it, so it is downloaded from the Hugging Face Hub.

1. Upload `web/models/conlang-gym-2b/` to a model repository, e.g. `hf upload <user>/<repo> web/models/conlang-gym-2b .`.
2. Point the site at it: `VITE_MODEL_URL=https://huggingface.co/<user>/<repo>/resolve/main/` at build time (the default is in `src/ui/model-store.ts`).
3. `.github/workflows/pages.yml` builds and publishes the site on GitHub Pages on every push to `main` that touches `web/`. Enable Pages with "GitHub Actions" as the source. Set `VITE_MODEL_URL` as a repository variable if it differs from the default.
