# large datasets near the end
galaxea_valid_datasets=(
    "part1_r1_lite"
    "part2_r1_lite"
    "part3_r1_lite"
    "part4_r1_lite"
    "part5_r1_lite"
)

for dataset_name in ${galaxea_valid_datasets[@]}; do
    echo "Generating dataset: $dataset_name"
    echo "=" * 100
    uv run python -m dataset_upload.generate_hf_dataset \
        --config_path=dataset_upload/configs/data_gen_configs/galaxea.yaml \
        --output.output_dir ./rfm_dataset/galaxea_rfm \
        --dataset.dataset_name galaxea_$dataset_name \

    echo "Done generating dataset: $dataset_name"
    echo "=" * 100
done