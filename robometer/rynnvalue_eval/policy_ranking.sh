#!/usr/bin/env bash
# RynnValue policy ranking on RBM-EVAL-OOD: does the value model rank better
# policies higher?
#
# Prerequisites (once):
#   export ROBOMETER_PROCESSED_DATASETS_PATH=/abs/path/to/processed_datasets
#   ./scripts/download_processed_datasets.sh && ./scripts/untar_processed_datasets.sh
#
# Reproduce the paper numbers for the released RynnValue-8B:
#   bash rynnvalue_eval/policy_ranking.sh
#
# The three inference attention masks behind the README table
# ("Policy Ranking: Attention Mask and Value Tokenizer") are selected by the
# first argument:
#   sdpa (default)      causal mask through the SDPA kernel   reported configuration
#   eager               causal mask through the eager kernel  same mask, slower kernel
#   isolated            training-time value-isolation mask (pred_slot_isolated_eager)
# e.g. bash rynnvalue_eval/policy_ranking.sh isolated
#
# State attn_implementation explicitly: left unset the run takes whatever the
# checkpoint's own config.json records, which is not the same across checkpoint
# generations.

# Resolve relative paths (robometer/evals/..., ./extracted_meta_with_descriptions.json)
# against this script's parent directory so the command runs from any cwd.
cd "$(dirname "$0")/.." || exit 1

VARIANT="${1:-sdpa}"
case "$VARIANT" in
    isolated) ATTN_OVERRIDE=("model_config.attn_implementation=pred_slot_isolated_eager") ;;
    eager|sdpa) ATTN_OVERRIDE=("model_config.attn_implementation=${VARIANT}") ;;
    *)
        echo "Unknown variant: ${VARIANT} (expected isolated, eager, or sdpa)" >&2
        exit 1
        ;;
esac

# Expected Kendall's tau (last frame) for the 8B quantile checkpoint, per suite,
# matching the README table:
#                        sdpa / eager (causal)   isolated
#   usc_franka                     0.750          0.667
#   usc_koch_p_ranking_all (rfm)   0.504          0.437
#   usc_trossen                    1.000          0.972
#   usc_xarm                       0.722          0.556
#   rfm_new_mit_franka             0.450          0.442
#   utd_so101_clean_top            0.800          0.800
#   mean                           0.704          0.646
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=Alibaba-DAMO-Academy/RynnValue-8B' \
    "output_dir=./baseline_eval_output/RynnValue-8B_policy_ranking_${VARIANT}" \
    'custom_eval.eval_types=[policy_ranking]' \
    'custom_eval.policy_ranking=[rbm-1m-ood]' \
    'custom_eval.use_frame_steps=false' \
    'custom_eval.pad_frames=false' \
    'custom_eval.num_examples_per_quality_pr=1000' \
    'max_frames=8' \
    'model_config.checkpoint_path=null' \
    'model_config.mode=absolute' \
    'model_config.stride=2' \
    'model_config.num_frames=8' \
    'model_config.camera_desc_lookup_path=./extracted_meta_with_descriptions.json' \
    "${ATTN_OVERRIDE[@]}"

# ---------------------------------------------------------------------------
# Templates for evaluating your own export instead of the released checkpoint.
# Fill in the paths, uncomment, and run.
# ---------------------------------------------------------------------------

# Exported HuggingFace model dir
# CUDA_VISIBLE_DEVICES=0 \
# HF_ENDPOINT=https://hf-mirror.com \
# python robometer/evals/run_baseline_eval.py \
#     'reward_model=rynnvalue' \
#     'model_path=/path/to/rynn_value_hf_model' \
#     'custom_eval.eval_types=[policy_ranking]' \
#     'custom_eval.policy_ranking=[rbm-1m-ood]' \
#     'custom_eval.use_frame_steps=false' \
#     'custom_eval.pad_frames=false' \
#     'custom_eval.num_examples_per_quality_pr=1000' \
#     'max_frames=64'

# HuggingFace model dir + raw training checkpoint (model.pt whose sibling
# huggingface/ holds the processor/config artifacts)
# CUDA_VISIBLE_DEVICES=0 \
# HF_ENDPOINT=https://hf-mirror.com \
# python robometer/evals/run_baseline_eval.py \
#     'reward_model=rynnvalue' \
#     'model_path=/path/to/rynn_value_hf_model' \
#     'custom_eval.eval_types=[policy_ranking]' \
#     'custom_eval.policy_ranking=[rbm-1m-ood]' \
#     'custom_eval.use_frame_steps=false' \
#     'custom_eval.pad_frames=false' \
#     'custom_eval.num_examples_per_quality_pr=1000' \
#     'max_frames=8' \
#     'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX'

# Raw training checkpoint only, with explicit output_dir
# CUDA_VISIBLE_DEVICES=0 \
# HF_ENDPOINT=https://hf-mirror.com \
# python robometer/evals/run_baseline_eval.py \
#     'reward_model=rynnvalue' \
#     'output_dir=./baseline_eval_output/rynnvalue/checkpoint_model_XXXXXX/' \
#     'custom_eval.eval_types=[policy_ranking]' \
#     'custom_eval.policy_ranking=[rbm-1m-ood]' \
#     'custom_eval.use_frame_steps=false' \
#     'custom_eval.pad_frames=false' \
#     'custom_eval.num_examples_per_quality_pr=1000' \
#     'max_frames=64' \
#     'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX'
