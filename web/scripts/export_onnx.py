# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "onnx", "huggingface_hub"]
# ///
"""Export the trained model to ONNX for the website (q4f16, text only).

The graph is onnx-community/Qwen3.5-2B-ONNX-OPT's text part: embed_tokens_q4f16 and
decoder_model_merged_q4f16, which run Gated DeltaNet with the LinearAttention and
CausalConvWithState kernels of onnxruntime-web's WebGPU backend. The vision encoder is left out.
Every weight in the two graphs is replaced by the checkpoint's, converted and quantized the way the
reference files were built: 4-bit MatMulNBits / GatherBlockQuantized in blocks of 32 with zero
points, RMSNorm weights stored as 1 + w, the gated norm rounded to bfloat16, A as -exp(A_log).

--check rebuilds the reference files themselves from their source checkpoint (Qwen/Qwen3.5-2B) and
compares every initializer, which shows that the mapping and the quantization are right.

Usage (from the repository root):
    uv run web/scripts/export_onnx.py --out_dir web/models/conlang-gym-2b
    uv run web/scripts/export_onnx.py --check
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download
from onnx import numpy_helper

logger = logging.getLogger(__name__)

GRAPH_REPO = "onnx-community/Qwen3.5-2B-ONNX-OPT"
GRAPH_SOURCE_REPO = "Qwen/Qwen3.5-2B"
MODEL_REPO = "nagohachi/Qwen3.5-2B-conlang-gym-v1-word-order-with-dictionary"
ONNX_FILES = ("embed_tokens_q4f16.onnx", "decoder_model_merged_q4f16.onnx")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
BLOCK_SIZE = 32
PREFIX = "model.language_model."

# Quantized weights: ONNX initializer prefix (layer index as {i}) -> checkpoint name.
QUANTIZED = {
    "model_layers_{i}_gdn_in_proj_qkv_MatMul_weight": "layers.{i}.linear_attn.in_proj_qkv.weight",
    "model_layers_{i}_gdn_in_proj_z_MatMul_weight": "layers.{i}.linear_attn.in_proj_z.weight",
    "model_layers_{i}_gdn_in_proj_a_MatMul_weight": "layers.{i}.linear_attn.in_proj_a.weight",
    "model_layers_{i}_gdn_in_proj_b_MatMul_weight": "layers.{i}.linear_attn.in_proj_b.weight",
    "model_layers_{i}_gdn_out_proj_MatMul_weight": "layers.{i}.linear_attn.out_proj.weight",
    "model_layers_{i}_attn_q_proj_MatMul_weight": "layers.{i}.self_attn.q_proj.weight",
    "model_layers_{i}_attn_k_proj_MatMul_weight": "layers.{i}.self_attn.k_proj.weight",
    "model_layers_{i}_attn_v_proj_MatMul_weight": "layers.{i}.self_attn.v_proj.weight",
    "model_layers_{i}_attn_o_proj_MatMul_weight": "layers.{i}.self_attn.o_proj.weight",
    "model_layers_{i}_mlp_gate_proj_MatMul_weight": "layers.{i}.mlp.gate_proj.weight",
    "model_layers_{i}_mlp_up_proj_MatMul_weight": "layers.{i}.mlp.up_proj.weight",
    "model_layers_{i}_mlp_down_proj_MatMul_weight": "layers.{i}.mlp.down_proj.weight",
    # Tied embeddings: the lm_head is the embedding matrix.
    "lm_head_MatMul_weight": "embed_tokens.weight",
    "model_embed_tokens_weight": "embed_tokens.weight",
}


def bfloat16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), returned as float32."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32)


def plus_one(x: np.ndarray) -> np.ndarray:
    """Return 1 + w: Qwen3.5's RMSNorm scales by 1 + weight, the ONNX op by weight."""
    return 1 + x


def neg_exp(x: np.ndarray) -> np.ndarray:
    """Return -exp(A_log), the decay rate A of the Gated DeltaNet."""
    return -np.exp(x)


# Unquantized weights: ONNX initializer name (layer index as {i}) -> checkpoint name, conversion.
DIRECT: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "model.layers.{i}.input_layernorm.weight": ("layers.{i}.input_layernorm.weight", plus_one),
    "model.layers.{i}.post_attention_layernorm.weight": (
        "layers.{i}.post_attention_layernorm.weight",
        plus_one,
    ),
    "model.layers.{i}.final_norm_layernorm.weight": ("norm.weight", plus_one),
    "model.layers.{i}.attn.q_norm.layernorm.weight": (
        "layers.{i}.self_attn.q_norm.weight",
        plus_one,
    ),
    "model.layers.{i}.attn.k_norm.layernorm.weight": (
        "layers.{i}.self_attn.k_norm.weight",
        plus_one,
    ),
    "model.layers.{i}.gdn.norm.weight": ("layers.{i}.linear_attn.norm.weight", bfloat16_round),
    "model.layers.{i}.gdn.A_neg_exp": ("layers.{i}.linear_attn.A_log", neg_exp),
    "model.layers.{i}.gdn.dt_bias": ("layers.{i}.linear_attn.dt_bias", lambda x: x),
    "model.layers.{i}.gdn.conv1d.weight_3d": ("layers.{i}.linear_attn.conv1d.weight", lambda x: x),
}


class SafetensorsReader:
    """Read tensors of a .safetensors file as float32 without torch (bfloat16 is widened)."""

    def __init__(self, path: Path) -> None:
        """Open path and read its header."""
        self.file = path.open("rb")
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(header_size))
        self.data_start = 8 + header_size

    def get(self, name: str) -> np.ndarray:
        """Return the tensor called name as a float32 array."""
        meta = self.header[name]
        start, end = meta["data_offsets"]
        self.file.seek(self.data_start + start)
        raw = self.file.read(end - start)
        if meta["dtype"] == "BF16":
            array = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] == "F32":
            array = np.frombuffer(raw, np.float32)
        else:
            raise ValueError(f"{name}: unsupported dtype {meta['dtype']}")
        return array.reshape(meta["shape"])


def quantize_q4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weight [N, K] to 4 bits per block of 32 along K, with a zero point per block.

    Return the packed values [N, K/32, 16] (two per byte, low nibble first), the float16 scales
    [N, K/32] and the packed zero points [N, ceil(K/64)]. A block's range always includes 0. A block
    whose scale underflows float16 gets scale 1 and zero point 0, like the reference files.
    """
    rows, cols = weight.shape
    num_blocks = cols // BLOCK_SIZE
    packed = np.empty((rows, num_blocks, BLOCK_SIZE // 2), np.uint8)
    scales = np.empty((rows, num_blocks), np.float16)
    zero_points = np.zeros((rows, (num_blocks + 1) // 2), np.uint8)
    chunk = 16384
    for start in range(0, rows, chunk):
        blocks = (
            weight[start : start + chunk].astype(np.float32).reshape(-1, num_blocks, BLOCK_SIZE)
        )
        low = np.minimum(blocks.min(-1), 0)
        high = np.maximum(blocks.max(-1), 0)
        scale = (high - low) / np.float32(15)
        scale = np.where(scale.astype(np.float16) == 0, np.float32(1), scale)
        zero_point = np.clip(np.round(-low / scale), 0, 15)
        values = np.round(blocks / scale[..., None] + zero_point[..., None])
        values = np.clip(values, 0, 15).astype(np.uint8)
        packed[start : start + chunk] = values[..., 0::2] | (values[..., 1::2] << 4)
        scales[start : start + chunk] = scale.astype(np.float16)
        zero_point = zero_point.astype(np.uint8)
        zero_points[start : start + chunk, :] = zero_point[:, 0::2]
        zero_points[start : start + chunk, : num_blocks // 2] |= zero_point[:, 1::2] << 4
    return packed, scales, zero_points


def build_initializer(name: str, reader: SafetensorsReader, cache: dict) -> np.ndarray | None:
    """Return the new value of the ONNX initializer name, or None if it is not a weight."""
    for onnx_pattern, checkpoint_pattern in QUANTIZED.items():
        regex = re.escape(onnx_pattern).replace(r"\{i\}", r"(\d+)") + r"_(quant|scales|zp)$"
        match = re.fullmatch(regex, name)
        if match:
            *layer, part = match.groups()
            checkpoint_name = PREFIX + checkpoint_pattern.format(i=layer[0] if layer else "")
            if checkpoint_name not in cache:
                cache.clear()  # keep at most one quantized matrix in memory
                cache[checkpoint_name] = quantize_q4(reader.get(checkpoint_name))
            packed, scales, zero_points = cache[checkpoint_name]
            return {"quant": packed, "scales": scales, "zp": zero_points}[part]
    for onnx_pattern, (checkpoint_pattern, convert) in DIRECT.items():
        regex = re.escape(onnx_pattern).replace(r"\{i\}", r"(\d+)")
        match = re.fullmatch(regex, name)
        if match:
            checkpoint_name = PREFIX + checkpoint_pattern.format(i=match.group(1))
            return convert(reader.get(checkpoint_name).astype(np.float32))
    return None


def replace_weights(
    graph_path: Path, reader: SafetensorsReader
) -> tuple[onnx.ModelProto, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load the ONNX file and replace every weight initializer with the checkpoint's.

    Return the model and, per replaced initializer, the (old, new) values.
    """
    model = onnx.load(graph_path)
    cache: dict = {}
    replaced = {}
    for index, tensor in enumerate(model.graph.initializer):
        new = build_initializer(tensor.name, reader, cache)
        if new is None:
            if not tensor.name.startswith("/model/constants/") and tensor.name != "model.inv_freq":
                raise ValueError(f"{graph_path.name}: no checkpoint weight for {tensor.name}")
            continue
        old = numpy_helper.to_array(tensor)
        new = new.reshape(old.shape).astype(old.dtype)
        replaced[tensor.name] = (old, new)
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(new, tensor.name))
    return model, replaced


def find_checkpoint(repo_dir: Path) -> Path:
    """Return the single .safetensors file of a downloaded checkpoint."""
    files = sorted(repo_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"expected one .safetensors file in {repo_dir}, found {files}")
    return files[0]


def resolve(repo_or_dir: str, allow_patterns: list[str] | None = None) -> Path:
    """Return repo_or_dir if it is a local directory, else download that Hub repository."""
    path = Path(repo_or_dir)
    if path.is_dir():
        return path
    return Path(snapshot_download(repo_or_dir, allow_patterns=allow_patterns))


def check(graph_dir: Path, source_dir: Path) -> None:
    """Rebuild the reference graph's weights from its source checkpoint and compare them."""
    reader = SafetensorsReader(find_checkpoint(source_dir))
    for file_name in ONNX_FILES:
        _, replaced = replace_weights(graph_dir / "onnx" / file_name, reader)
        mismatched = []
        for name, (old, new) in replaced.items():
            if np.array_equal(old, new):
                continue
            # -exp() may differ in the last bit between numpy and torch.
            ulps = np.abs(old.view(np.int32).astype(np.int64) - new.view(np.int32)).max()
            if name.endswith("A_neg_exp") and ulps <= 1:
                continue
            mismatched.append(name)
        logger.info(
            "%s: %d initializers rebuilt, %d differ%s",
            file_name,
            len(replaced),
            len(mismatched),
            f": {mismatched[:10]}" if mismatched else "",
        )
        if mismatched:
            raise SystemExit(1)


def export(graph_dir: Path, model_dir: Path, out_dir: Path) -> None:
    """Write the ONNX files, tokenizer and configs of the website model to out_dir."""
    reader = SafetensorsReader(find_checkpoint(model_dir))
    (out_dir / "onnx").mkdir(parents=True, exist_ok=True)
    for file_name in ONNX_FILES:
        model, replaced = replace_weights(graph_dir / "onnx" / file_name, reader)
        target = out_dir / "onnx" / file_name
        for stale in (target, target.with_name(target.name + "_data")):
            stale.unlink(missing_ok=True)
        onnx.save_model(
            model,
            target,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=file_name + "_data",
        )
        logger.info("%s: replaced %d initializers", file_name, len(replaced))
    for file_name in TOKENIZER_FILES:
        shutil.copy(model_dir / file_name, out_dir / file_name)
    shutil.copy(model_dir / "generation_config.json", out_dir / "generation_config.json")
    # The graph's config carries transformers.js_config; the architecture is the checkpoint's.
    shutil.copy(graph_dir / "config.json", out_dir / "config.json")
    write_manifest(out_dir)


def write_manifest(out_dir: Path) -> None:
    """Write manifest.json: each file's size, and a version that changes with any file's content.

    The website reads the sizes for its progress bar and appends the version to the file URLs, so a
    re-exported model is not served from the browser's cache.
    """
    files = sorted(
        p
        for p in out_dir.rglob("*")
        if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(out_dir)).encode() + b"\0")
        with path.open("rb") as file:
            digest.update(hashlib.file_digest(file, "sha256").digest())
    manifest = {
        "model": MODEL_REPO,
        "graph": GRAPH_REPO,
        "version": digest.hexdigest()[:12],
        "files": {str(p.relative_to(out_dir)): p.stat().st_size for p in files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(manifest["files"].values())
    logger.info("wrote %s (%.1f MB, version %s)", out_dir, total / 1e6, manifest["version"])


def main() -> None:
    """Export the website model, or check the conversion against the reference files."""
    parser = argparse.ArgumentParser(description="Export the conlang-gym model to ONNX (q4f16).")
    parser.add_argument("--out_dir", type=Path, default=Path("web/models/conlang-gym-2b"))
    parser.add_argument("--model", default=MODEL_REPO, help="Hub id or local checkpoint directory")
    parser.add_argument("--graph", default=GRAPH_REPO, help="Hub id or local directory")
    parser.add_argument("--check", action="store_true", help="verify against the reference files")
    parser.add_argument(
        "--check_source", default=GRAPH_SOURCE_REPO, help="the checkpoint the graph was built from"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    patterns = ["*.json", "*.jinja", *(f"onnx/{name}*" for name in ONNX_FILES)]
    graph_dir = resolve(args.graph, patterns)
    if args.check:
        check(graph_dir, resolve(args.check_source, ["*.safetensors"]))
        return
    export(graph_dir, resolve(args.model), args.out_dir)


if __name__ == "__main__":
    main()
