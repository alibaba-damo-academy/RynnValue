# large datasets near the end
OXE_VALID_DATASETS=(
    "austin_buds_dataset_converted_externally_to_rlds"
    "austin_sirius_dataset_converted_externally_to_rlds"
    "berkeley_cable_routing"
    "berkeley_fanuc_manipulation"
    "dlr_edan_shared_control_converted_externally_to_rlds"
    "fmb"
    "furniture_bench_dataset_converted_externally_to_rlds"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    "jaco_play"
    "stanford_hydra_dataset_converted_externally_to_rlds"
    "taco_play"
    "toto"
    "ucsd_kitchen_dataset_converted_externally_to_rlds"
    "utaustin_mutex"
    "viola"
    "fractal20220817_data"
    "bc_z"
    "language_table"
    "bridge_v2"
    "droid"
    # new datasets
    "robo_set"
    "aloha_mobile"
    "tidybot"
    "imperialcollege_sawyer_wrist_cam"
    "kaist_nonprehensile_converted_externally_to_rlds"
    "berkeley_mvp_converted_externally_to_rlds"
    "berkeley_rpt_converted_externally_to_rlds"
    "nyu_rot_dataset_converted_externally_to_rlds"
    "tokyo_u_lsmo_converted_externally_to_rlds"
)

for dataset_name in ${OXE_VALID_DATASETS[@]}; do
    echo "Generating dataset: $dataset_name"
    echo "=" * 100
    uv run python -m dataset_upload.generate_hf_dataset \
        --config_path=dataset_upload/configs/data_gen_configs/oxe.yaml \
        --output.output_dir ./robometer_dataset/oxe_rfm \
        --dataset.dataset_path ~/tensorflow_datasets/openx_datasets/ \
        --hub.push_to_hub=false \
        --dataset.dataset_name oxe_$dataset_name \

    echo "Done generating dataset: $dataset_name"
    echo "=" * 100
done