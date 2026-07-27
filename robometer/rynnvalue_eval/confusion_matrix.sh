# match_binary variant (default): confusion-matrix emits a bounded 0/1 match score
# (Match:Yes->1.0, else 0.0) instead of the -100 sentinel.
CUDA_VISIBLE_DEVICES=1 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=/path/to/rynn_value_hf_model' \
    'output_dir=baseline_eval_output/rynnvalue/confusion_matchscore_038000' \
    'custom_eval.eval_types=[confusion_matrix]' \
    'custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]]' \
    'max_frames=8' \
    'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX' \
    'model_config.confusion_score_mode=match_binary' \
    'save_videos=false'


# normalized-value variant: confusion-matrix uses the value head normalized to
# [0,1] (1 - t/t_max) on Match:Yes, else 0.0 (Match:No or no verdict). Still runs
# the Analysis. Select via model_config.confusion_score_mode.
CUDA_VISIBLE_DEVICES=1 \
HF_ENDPOINT=https://hf-mirror.com \
ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets \
python robometer/evals/run_baseline_eval.py \
    'reward_model=rynnvalue' \
    'model_path=/path/to/rynn_value_hf_model' \
    'output_dir=baseline_eval_output/rynnvalue/confusion_normvalue_038000' \
    'custom_eval.eval_types=[confusion_matrix]' \
    'custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]]' \
    'max_frames=8' \
    'model_config.checkpoint_path=/path/to/checkpoint_model_XXXXXX' \
    'model_config.confusion_score_mode=normalized_value' \
    'save_videos=false'