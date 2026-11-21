# conlang-gym

An RL environment in which a language model works as a field linguist: it questions an informant about a randomly generated artificial language, then translates meanings it has not seen into that language. 

Trained model: [nagohachi/Qwen3.5-2B-conlang-gym-v1-word-order-with-dictionary](https://huggingface.co/nagohachi/Qwen3.5-2B-conlang-gym-v1-word-order-with-dictionary)

## Layout

| Path | Contents |
|---|---|
| `conlang_gym/language.py` | Language generation, the informant, action parsing and scoring |
| `conlang_gym/solver.py` | Hypothesis solver |
| `conlang_gym/make_dataset.py` | Building dataset |
| `conlang_gym/agent_loop.py`, `agent_loop.yaml` | agent loop that runs one episode, with an optional thinking budget and coverage reward |
| `scripts/train_qwen3_5_2b.sh` | GRPO training |
| `scripts/train_qwen3_5_2b_plural_past.sh` | GRPO training on word order + plural + past |
| `scripts/eval_qwen3_5_2b_transfer.sh` | Validation-only evaluation on languages with phenomena absent from training |
| `scripts/analyze_validation.py` | Outputs the word orders the model wrote (from validation dumps) and its accuracy |
| `web/` | Website: play episodes yourself, or watch the trained model play them in the browser (WebGPU); see [web/README.md](web/README.md) |
| `libs/verl` | A copy of verl (see below) |

## Setup

Linux, an NVIDIA driver for CUDA 13.0 (580 or later) and [uv](https://docs.astral.sh/uv/). The sync builds `causal-conv1d` from source, so a CUDA 13 `nvcc` must be on `PATH` (or set `CUDA_HOME`).

```bash
cd libs/verl
uv sync --frozen --all-packages --extra vllm --extra fsdp
cd ../..
source libs/verl/.venv/bin/activate
```

## Data

```bash
python -m conlang_gym.make_dataset --out_dir outputs/data
```

This writes `train.parquet` (8,000 languages), `val_seen.parquet` (64 languages per training word order, with seeds not used in training) and `val_unseen.parquet` (64 languages per held-out word order). 
Each row stores only the seed and the environment settings, and the agent loop regenerates the language from them. 
Every field of `EnvConfig` in `conlang_gym/language.py` is also a flag, e.g. `--active_params past object_marking`, `--budget` or `--show_dictionary`.
`--active_params` takes any of `object_marking`, `plural`, `past`, `subject_agreement`, `vowel_harmony`, `ergative_marking`, `object_plural` and `verb_number_stem` (see `Grammar` in `conlang_gym/language.py`), and `--solver_params` sets the parameters the solver considers when it keeps only solvable seeds. 

## Training

```bash
bash scripts/train_qwen3_5_2b.sh nothink   # or: think, coverage
bash scripts/train_qwen3_5_2b_plural_past.sh curriculum   # or: lora, full
```
 
The published model is the `nothink` run after its 62 steps. 
Metrics go to wandb (project `conlang_gym`), and checkpoints, rollouts and validation dumps go to `outputs/`.
`coverage` adds 0.2 x the fraction of conditions the questions cover to the reward, and `curriculum` trains on word order + plural + past (`--active_params plural past --out_dir outputs/data/plural_past`) starting from the published model. 

## Evaluation

```bash
python scripts/analyze_validation.py outputs/val_generations/qwen3_5_2b_nothink
bash scripts/eval_qwen3_5_2b_transfer.sh a   # or: b
```

`eval_qwen3_5_2b_transfer.sh` compares the published model (a) with the `coverage` run (b) on languages with ergative marking, object plurals or verb number stems; its header lists the commands that build the test data. 

## Website

`web/` is a static site with a TypeScript port of the environment (same seeds, same languages) and the trained model, quantized to 4 bits and run on the visitor's GPU with onnxruntime-web. A person can play an episode, or let the model play it. See [web/README.md](web/README.md) for development, the model export and deployment.

## libs/verl

A copy of [verl](https://github.com/verl-project/verl) at commit `a4e0dd21`: the `verl` package and the files needed to build it. 

