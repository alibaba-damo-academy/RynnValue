uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_10_256 \
    --dataset.dataset_name=libero256_10 

uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_90_256 \
    --dataset.dataset_name=libero256_90 

uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_spatial_256 \
    --dataset.dataset_name=libero256_spatial 

uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_goal_256 \
    --dataset.dataset_name=libero256_goal 

uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_object_256 \
    --dataset.dataset_name=libero256_object