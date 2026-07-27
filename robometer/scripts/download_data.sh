#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Fast dataset setup via either Hugging Face CLI downloads (default)
# or git-lfs clones from the Hub.
#
#  * HF CLI path requires: `pip install huggingface_hub` (provides `hf` CLI)
#  * Git path requires: `git` and `git-lfs`
# ------------------------------------------------------------------

# ------------------------------
# Parse command‑line arguments
# ------------------------------
METHOD=${RFM_DOWNLOAD_METHOD:-hf}   # hf | git
BASE_DIR_DEFAULT=${ROBOMETER_DATASET_PATH:-${RFM_DATASET_PATH:-./robometer_dataset}}
BASE_DIR="$BASE_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git)           METHOD="git"; shift ;;
    --hf)            METHOD="hf"; shift ;;
    --method=*)      METHOD="${1#*=}"; shift ;;
    --method)        METHOD="$2"; shift 2 ;;
    --dir|--base-dir|-d)
                      BASE_DIR="$2"; shift 2 ;;
    *)               BASE_DIR="$1"; shift ;;
  esac
done

# ------------------------------
# Sanity checks for the chosen method
# ------------------------------
case "$METHOD" in
  hf)
    if ! command -v hf >/dev/null 2>&1; then
      echo "Error: 'hf' CLI not found. Install with:" >&2
      echo "   uv pip install huggingface_hub (or ensure your venv is activated)" >&2
      exit 1
    fi
    ;;
  git)
    if ! command -v git >/dev/null 2>&1; then
      echo "Error: git not found. Please install git." >&2
      exit 1
    fi
    if ! git lfs version >/dev/null 2>&1; then
      echo "Warning: git-lfs not found. You may end up with pointer files." >&2
      echo "   Install git-lfs for full downloads." >&2
    fi
    ;;
  *)
    echo "Error: Unknown METHOD='${METHOD}'. Use 'hf' or 'git'." >&2
    exit 1
    ;;
esac

mkdir -p "${BASE_DIR}"

# ------------------------------
# Helper to download a dataset repo
# ------------------------------
download_dataset() {
  local repo_id="$1"                # e.g., abraranwar/libero_rfm
  local name="${repo_id##*/}"       # last path segment as folder name
  local target_dir="${BASE_DIR}/${name}"

  echo "Downloading ${repo_id} -> ${target_dir} via ${METHOD}"
  if [[ "$METHOD" == "hf" ]]; then
    hf download "${repo_id}" \
      --repo-type dataset \
      --local-dir "${target_dir}"
  else
    local url="https://huggingface.co/datasets/${repo_id}.git"
    if [[ -d "${target_dir}/.git" ]]; then
      echo "Updating existing clone at ${target_dir}"
      git -C "${target_dir}" remote set-url origin "${url}" || true
      git -C "${target_dir}" fetch --all --tags
      git -C "${target_dir}" pull --ff-only
    else
      git clone "${url}" "${target_dir}"
    fi
    if git lfs version >/dev/null 2>&1; then
      git -C "${target_dir}" lfs install --local >/dev/null 2>&1 || true
      git -C "${target_dir}" lfs pull || true
    fi
  fi
}

# ------------------------------
# Retry helper – keeps trying until success
# ------------------------------
retry_until_success() {
  local cmd="$1"
  local desc="${2:-$cmd}"

  echo "=== ${desc} ==="
  until eval "$cmd"; do
    echo "❌ $desc failed – retrying in 30s …"
    sleep 30
  done
  echo "✅ $desc succeeded."
}

# ------------------------------
# List of all repos to download (including the ones that were previously commented out)
# ------------------------------
repos=(
  abraranwar/libero_rfm
  abraranwar/agibotworld_alpha_rfm   
  abraranwar/agibotworld_alpha_headcam_rfm   
  abraranwar/usc_koch_rewind_rfm
  ykorkmaz/libero_failure_rfm
  aliangdw/metaworld
  jesbu1/oxe_rfm
  jesbu1/galaxea_rfm
  jesbu1/molmoact_rfm
  jesbu1/ph2d_rfm
  jesbu1/epic_rfm
  jesbu1/failsafe_rfm
  jesbu1/h2r_rfm
  jesbu1/roboarena_0825_rfm
  jesbu1/oxe_rfm_eval
  anqil/rh20t_subset_rfm # can be replaced with anqil/rh20t_rfm full dataset
  jesbu1/humanoid_everyday_rfm
  jesbu1/motif_rfm
  jesbu1/auto_eval_rfm
  jesbu1/soar_rfm 
  jesbu1/racer_rfm
  jesbu1/egodex_rfm
  aliangdw/usc_xarm_policy_ranking
  aliangdw/usc_franka_policy_ranking
  aliangdw/utd_so101_policy_ranking
  aliangdw/utd_so101_human
  jesbu1/mit_franka_p-rank_rfm
  jesbu1/utd_so101_clean_policy_ranking_top
  jesbu1/utd_so101_clean_policy_ranking_wrist
  jesbu1/usc_koch_human_robot_paired
  jesbu1/usc_koch_p_ranking_rfm
  #jesbu1/roboreward_rfm
  jesbu1/roboreward_rfm_high_res
  jesbu1/rfm_new_mit_franka_rfm_nowrist
  ykorkmaz/usc_trossen_rfm
  aliangdw/robofac_rbm
)

# ------------------------------
# Download each repo with retry logic
# ------------------------------
for repo in "${repos[@]}"; do
  # Skip lines that are still commented out (start with '#')
  [[ "$repo" == \#* ]] && continue

  retry_until_success "download_dataset $repo" "$repo"
done

echo ""
echo "Done. Set ROBOMETER_DATASET_PATH=${BASE_DIR} for training/eval."
