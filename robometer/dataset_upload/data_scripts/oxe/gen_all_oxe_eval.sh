# comment out things which have no eval split
OXE_VALID_DATASETS=(
    #"austin_buds_dataset_converted_externally_to_rlds"
    #"austin_sirius_dataset_converted_externally_to_rlds"
    "berkeley_cable_routing"
    #"berkeley_fanuc_manipulation"
    #"dlr_edan_shared_control_converted_externally_to_rlds"
    #"fmb"
    #"furniture_bench_dataset_converted_externally_to_rlds"
    #"iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    "jaco_play"
    #"stanford_hydra_dataset_converted_externally_to_rlds"
    "taco_play"
    "toto"
    #"ucsd_kitchen_dataset_converted_externally_to_rlds"
    #"utaustin_mutex"
    "viola"
    #"fractal20220817_data"
    "bc_z"
    #"language_table"
    "bridge_v2"
    #"droid"
)

for dataset_name in ${OXE_VALID_DATASETS[@]}; do
    echo "Generating dataset: $dataset_name"
    echo "=" * 100
    uv run dataset_upload/generate_hf_dataset.py \
        --config_path=dataset_upload/configs/data_gen_configs/oxe.yaml \
        --output.output_dir ~/scratch_data/oxe_rfm_eval \
        --dataset.dataset_name oxe_$dataset_name\_eval \
        --hub.hub_repo_id oxe_rfm_eval

    echo "Done generating dataset: $dataset_name"
    echo "=" * 100
done