# RynnValue: exported HuggingFace model dir
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=/path/to/rynn_value_hf_model' \
    'custom_eval.eval_types=[policy_ranking]' \
    'custom_eval.policy_ranking=[rbm-1m-ood]' \
    'custom_eval.use_frame_steps=false' \
    'custom_eval.pad_frames=false' \
    'custom_eval.num_examples_per_quality_pr=1000' \
    'max_frames=64' \
    'model_config.batch_size=32'

# RynnValue-8B official release, eager attention (reproduces the legacy
# pre-isolation eval numbers)
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=Alibaba-DAMO-Academy/RynnValue-8B' \
    'output_dir=./baseline_eval_output/RynnValue-8B_eager' \
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
    'model_config.attn_implementation=eager'

# RynnValue: HuggingFace model dir + raw training checkpoint (model.pt)
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=/path/to/rynn_value_hf_model' \
    'custom_eval.eval_types=[policy_ranking]' \
    'custom_eval.policy_ranking=[rbm-1m-ood]' \
    'custom_eval.use_frame_steps=false' \
    'custom_eval.pad_frames=false' \
    'custom_eval.num_examples_per_quality_pr=1000' \
    'max_frames=8' \
    'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX'

# RynnValue: raw training checkpoint only, with explicit output_dir
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'output_dir=./baseline_eval_output/rynnvalue/checkpoint_model_XXXXXX/' \
    'custom_eval.eval_types=[policy_ranking]' \
    'custom_eval.policy_ranking=[rbm-1m-ood]' \
    'custom_eval.use_frame_steps=false' \
    'custom_eval.pad_frames=false' \
    'custom_eval.num_examples_per_quality_pr=1000' \
    'max_frames=64' \
    'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX'
