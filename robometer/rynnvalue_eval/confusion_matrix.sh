# RynnValue-8B official release, match_binary scoring
# (bounded 0/1 match score: Match:Yes->1.0, else 0.0)
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=Alibaba-DAMO-Academy/RynnValue-8B' \
    'output_dir=./baseline_eval_output/rynnvalue/confusion_matchscore' \
    'custom_eval.eval_types=[confusion_matrix]' \
    'custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]]' \
    'max_frames=8' \
    'model_config.checkpoint_path=null' \
    'model_config.mode=absolute' \
    'model_config.stride=2' \
    'model_config.num_frames=8' \
    'model_config.camera_desc_lookup_path=./extracted_meta_with_descriptions.json' \
    'model_config.confusion_score_mode=match_binary' \
    'model_config.attn_implementation=eager' \
    'save_videos=false'

# RynnValue-8B official release, normalized_value scoring
# (value head normalized to [0,1] (1 - t/t_max) on Match:Yes, else 0.0)
CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=Alibaba-DAMO-Academy/RynnValue-8B' \
    'output_dir=./baseline_eval_output/rynnvalue/confusion_normvalue' \
    'custom_eval.eval_types=[confusion_matrix]' \
    'custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]]' \
    'max_frames=8' \
    'model_config.checkpoint_path=null' \
    'model_config.mode=absolute' \
    'model_config.stride=2' \
    'model_config.num_frames=8' \
    'model_config.camera_desc_lookup_path=./extracted_meta_with_descriptions.json' \
    'model_config.confusion_score_mode=normalized_value' \
    'model_config.attn_implementation=eager' \
    'save_videos=false'
