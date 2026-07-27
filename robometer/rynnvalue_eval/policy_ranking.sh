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
    'model_config.batch_size=32' \
    'model_config.conversation_type=Progress'

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
    'max_frames=64' \
    'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX' \
    'model_config.conversation_type=Progress'

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
    'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX' \
    'model_config.conversation_type=Progress'

# Robometer-4B baseline
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=/path/to/Robometer-4B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=32 \
    model_config.batch_size=4

# RoboReward-8B baseline
CUDA_VISIBLE_DEVICES=1 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=/path/to/RoboReward-8B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64

# Robo-Dopamine 8B baseline
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=/path/to/Robo-Dopamine-GRM-2.0-8B-Preview \
    model_config.eval_mode=forward \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64 \
    model_config.batch_size=1
