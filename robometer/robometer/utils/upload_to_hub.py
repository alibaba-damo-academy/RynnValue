#!/usr/bin/env python3
"""
Simple script to upload a trained model to HuggingFace Hub.

uv run python robometer/utils/upload_to_hub.py \
    --model_dir ./checkpoints/my_model \
    --hub_model_id rewardfm/my-rbm-model \
    --private \
    --commit_message "Upload trained RBM model checkpoint" \
    --base_model "Qwen/Qwen3-VL-4B-Instruct"
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, login


def validate_model_directory(model_dir: Path) -> bool:
    """Validate that the directory contains a valid model."""
    required_files = ["config.json"]

    # Check for required files
    for file in required_files:
        if not (model_dir / file).exists():
            print(f"❌ ERROR: Required file {file} not found in {model_dir}")
            return False

    # Check for model files (either single or sharded)
    has_model_files = False

    # Check for safetensors files
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if safetensors_files:
        has_model_files = True
        print(f"Found {len(safetensors_files)} safetensors files")

    # Check for pytorch files
    pytorch_files = list(model_dir.glob("pytorch_model*.bin"))
    if pytorch_files:
        has_model_files = True
        print(f"Found {len(pytorch_files)} pytorch files")

    if not has_model_files:
        print("❌ ERROR: No model files found (*.safetensors or pytorch_model*.bin)")
        return False

    print("✅ Model directory validation passed")
    return True


def create_model_card(model_dir: Path, base_model: str, model_name: str):
    """Create or update the model card."""
    readme_path = model_dir / "README.md"

    # Try to read existing config to get more info
    config_path = model_dir / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "unknown")
        config.get("architectures", ["unknown"])
    except:
        model_type = "unknown"

    # Try to read wandb info if available
    wandb_info_path = model_dir.parent / "wandb_info.json"
    wandb_section = ""
    if wandb_info_path.exists():
        try:
            with open(wandb_info_path) as f:
                wandb_info = json.load(f)
            wandb_notes = wandb_info.get("wandb_notes", "")
            notes_section = f"\n- **Notes**: {wandb_notes}" if wandb_notes else ""
            wandb_section = f"""
## Training Run

- **Wandb Run**: [{wandb_info.get("wandb_name", "N/A")}]({wandb_info.get("wandb_url", "#")})
- **Wandb ID**: `{wandb_info.get("wandb_id", "N/A")}`
- **Project**: {wandb_info.get("wandb_project", "N/A")}{notes_section}
"""
        except Exception as e:
            print(f"⚠️ WARNING: Could not read wandb info: {e}")
            wandb_section = ""

    model_card_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- reward_model
- rbm
- preference_comparisons
library_name: transformers
---

# {model_name}

## Model Details

- **Base Model**: {base_model}
- **Model Type**: {model_type}
{wandb_section}
## Citation

If you use this model, please cite:
"""

    with open(readme_path, "w") as f:
        f.write(model_card_content)

    print("Created/updated model card (README.md)")


def upload_model_to_hub(
    model_dir: str,
    hub_model_id: str,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload model",
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    tag_name: str | None = None,
):
    """
    Upload model directory to HuggingFace Hub.

    Args:
        model_dir: Path to the model directory
        hub_model_id: HuggingFace model ID (username/model-name)
        private: Whether to make the model private
        token: HuggingFace token
        commit_message: Commit message for the upload
        base_model: Base model name for the model card
        tag_name: Optional tag name to create after upload

    Returns:
        tuple: (hub_url, commit_id) - URL of the uploaded model and the commit ID
    """

    model_path = Path(model_dir)

    # Validate model directory
    if not model_path.exists():
        raise ValueError(f"Model directory does not exist: {model_path}")

    if not validate_model_directory(model_path):
        raise ValueError("Model directory validation failed")

    # Create/update model card
    create_model_card(model_path, base_model, hub_model_id)

    # Login to HuggingFace
    if token:
        login(token=token)
        print("Logged in to HuggingFace Hub")
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
        print("Logged in using HF_TOKEN environment variable")
    else:
        print("⚠️ WARNING: No HuggingFace token provided. You may need to login manually.")

    # Upload to Hub
    print(f"Uploading model to: {hub_model_id}")
    print(f"Private: {private}")
    print(f"Model directory: {model_path}")

    api = HfApi()

    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=hub_model_id, repo_type="model", private=private, exist_ok=True)
        print(f"Repository {hub_model_id} created/verified")
    except Exception as e:
        print(f"⚠️ WARNING: Could not create repository (may already exist): {e}")

    # Upload the entire directory
    commit_info = api.upload_folder(
        folder_path=str(model_path), repo_id=hub_model_id, commit_message=commit_message, repo_type="model"
    )

    commit_id = commit_info.oid
    print(f"✅ Successfully uploaded model to: https://huggingface.co/{hub_model_id}")
    print(f"📋 Commit ID: {commit_id}")

    # Also upload the config.yaml which is in the directory above
    config_path = model_path.parent / "config.yaml"
    if config_path.exists():
        print(f"Uploading config.yaml to: {hub_model_id}")
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.yaml",
            repo_id=hub_model_id,
            repo_type="model",
        )

    # Create tag if requested
    if tag_name:
        api.create_tag(
            repo_id=hub_model_id, repo_type="model", tag=tag_name, revision=commit_id, tag_message=commit_message
        )
        print(f"🏷️ Created tag: {tag_name}")

    hub_url = f"https://huggingface.co/{hub_model_id}"
    return hub_url, commit_id


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--hub_model_id", type=str, required=True, help="HuggingFace model ID (username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make the model private")
    parser.add_argument("--token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--commit_message", type=str, default="Upload RBM model", help="Commit message for the upload")
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name for the model card"
    )

    args = parser.parse_args()

    try:
        url, commit_id = upload_model_to_hub(
            model_dir=args.model_dir,
            hub_model_id=args.hub_model_id,
            private=args.private,
            token=args.token,
            commit_message=args.commit_message,
            base_model=args.base_model,
        )

        print("\n🎉 Upload completed successfully!")
        print(f"Model URL: {url}")
        print(f"Commit ID: {commit_id}")

    except Exception as e:
        print(f"❌ ERROR: Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
