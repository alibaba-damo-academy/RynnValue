DATA_SOURCE_ROBOT_DESCRIPTION = {
    "usc_koch_p_ranking_all": "a Koch dual-arm robot",
    "rfm_new_mit_franka_rfm": "a Franka single-arm robot",
    "usc_franka_policy_ranking": "a Franka single-arm robot",
    "usc_trossen": "an Trossen dual-arm robot",
    "usc_xarm_policy_ranking": "an xArm single-arm robot",
    "utd_so101_clean_policy_ranking_top": "an SO-101 single-arm robot",
}

DATA_SOURCE_CAMERA_DESCRIPTION = {
    "usc_koch_p_ranking_all": "the top-down camera",
    "rfm_new_mit_franka_rfm": None,  # per-trajectory: "the wrist-mounted camera" or "the main camera", stored in metadata
    "usc_franka_policy_ranking": "the main camera",
    "usc_trossen": "the left wrist-mounted camera",
    "usc_xarm_policy_ranking": "the main right camera",
    "utd_so101_clean_policy_ranking_top": "the main left camera",
}
