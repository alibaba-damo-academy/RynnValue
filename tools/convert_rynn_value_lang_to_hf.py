import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import argparse
import importlib
import shutil
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel, AutoProcessor


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a RynnValueLang checkpoint and re-export it in Hugging Face format."
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        help="Path to a `checkpoint_model_*` directory containing `model.pt`. "
             "The sibling `huggingface/` directory is used for config/processor/source files.", 
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to write the Hugging Face-format model.",
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default=None,
        help="Override path to the source `huggingface/` directory. "
             "Defaults to `<model_ckpt_path>/../huggingface`.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=list(DTYPE_MAP.keys()),
        help="Dtype to cast weights to before saving.",
    )
    parser.add_argument(
        "--use_local_code",
        action="store_true",
        default=False,
        help="Build/load the model with the local `rynn_value/` package (registered "
             "on import) instead of the `trust_remote_code` .py bundled in `hf_path`, "
             "and export those local sources into the output model.",
    )
    parser.add_argument(
        "--local_code_path",
        type=str,
        default=None,
        help="Path to the local `rynn_value/` package used with --use_local_code. "
             "Defaults to `<repo_root>/rynn_value`.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.model_ckpt_path)
    hf_path = Path(args.hf_path) if args.hf_path else ckpt_path.parent / "huggingface"
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    dtype = DTYPE_MAP[args.dtype]

    local_code_path = None
    if args.use_local_code:
        local_code_path = (
            Path(args.local_code_path) if args.local_code_path
            else Path(parent_dir) / "rynn_value"
        )
        if not local_code_path.is_dir():
            raise FileNotFoundError(f"local_code_path not found: {local_code_path}")
        pkg_parent = str(local_code_path.resolve().parent)
        if pkg_parent not in sys.path:
            sys.path.insert(0, pkg_parent)
        print(f"Registering local RynnValueLang classes from {local_code_path}")
        importlib.import_module(local_code_path.name)
    # With the local package imported the classes are registered, so load them
    # directly (trust_remote_code=False) rather than the .py bundled in hf_path.
    trust_remote_code = not args.use_local_code

    print(f"Loading processor/config from {hf_path}")
    processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=trust_remote_code)
    hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=trust_remote_code)

    print("Building empty model from config...")
    with init_empty_weights():
        model = AutoModel.from_config(hf_config, trust_remote_code=trust_remote_code)

    state_dict_path = ckpt_path / "model.pt"
    print(f"Loading state dict from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    # Re-tie shared weights (e.g. Qwen3VL's lm_head.weight <- embed_tokens.weight).
    # After `assign=True` load, tied parameters that were absent from the state dict
    # remain as meta tensors; tie_weights() rewires them to the loaded embedding storage.
    model.tie_weights()

    # Materialize any remaining meta params/buffers (should be none for a valid ckpt).
    remaining_meta = [
        name for name, p in model.named_parameters() if p.is_meta
    ] + [
        name for name, b in model.named_buffers() if b.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(
            f"The following parameters/buffers are still on meta after loading "
            f"and tie_weights(): {remaining_meta}. The checkpoint is missing "
            "weights that are not covered by weight tying."
        )

    model = model.to(dtype=dtype)
    model.eval()

    print(f"Saving Hugging Face model to {output_path}")
    model.save_pretrained(output_path)

    # Overlay the snapshot `huggingface/` dir (processor files, custom source .py,
    # auto_map-patched config.json) so the exported checkpoint matches the source
    # files captured at training time rather than whatever is on disk now.
    print(f"Overlaying source files from {hf_path}")
    shutil.copytree(hf_path, output_path, dirs_exist_ok=True)

    # In local-code mode, overwrite the snapshot's custom .py with the local
    # package's sources so the exported model runs the local code. Non-code files
    # (tokenizer, auto_map-patched config.json, chat template) still come from hf_path.
    if args.use_local_code:
        print(f"Overlaying local source .py from {local_code_path}")
        for py_file in sorted(local_code_path.glob("*.py")):
            shutil.copy2(py_file, output_path / py_file.name)

    print("Done.")


if __name__ == "__main__":
    main()
