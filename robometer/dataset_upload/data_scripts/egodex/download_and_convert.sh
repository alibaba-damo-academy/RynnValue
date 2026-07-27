for part in test part1 part2 part3 part4 part5; do

    echo "Processing ${part}..."

    # Download the dataset. 
    # Example curl "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip
    # Download it to ~/egodex/${part}.zip
    # check first if it doesn't exist
    mkdir -p ${HOME}/egodex
    if [ ! -d "${HOME}/egodex/${part}" ]; then
        echo "curl "https://ml-site.cdn-apple.com/datasets/egodex/${part}.zip" -o ${HOME}/egodex/${part}.zip"
        curl "https://ml-site.cdn-apple.com/datasets/egodex/${part}.zip" -o ${HOME}/egodex/${part}.zip
        unzip -d ${HOME}/egodex/${part} ${HOME}/egodex/${part}.zip
        rm ${HOME}/egodex/${part}.zip
    fi

    uv run python -m dataset_upload.generate_hf_dataset \
        --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml \
        --dataset.dataset_path="${HOME}/egodex/${part}/${part}" \
        --dataset.dataset_name="egodex_${part}" \
        --output.output_dir=./robometer_dataset/egodex_rfm \
        --hub.push_to_hub=false

    echo "Done processing ${part}..."

    # Delete the dataset
    echo "Deleting ${HOME}/egodex/${part}..."
    rm -rf ${HOME}/egodex/${part}
    echo "Done deleting ${HOME}/egodex/${part}..."

done