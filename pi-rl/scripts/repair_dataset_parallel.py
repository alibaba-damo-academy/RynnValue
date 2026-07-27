#!/usr/bin/env python3
# adapted from openpi
"""
Data repair script - write to tmp first then move.
"""

import os
import shutil
import argparse
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path


def detect_jumps(episode_df, task_type, jump_threshold=0.15):
    if 'action.arm' not in episode_df.columns:
        return 0
    
    if task_type == 'single':
        indices = list(range(7))
    else:
        indices = list(range(14))
    
    jump_counts = []
    for idx in indices:
        values = episode_df['action.arm'].apply(lambda arr: arr[idx] if isinstance(arr, np.ndarray) and len(arr) > idx else np.nan)
        diff = values.diff().abs()
        jumps = diff > jump_threshold
        jump_counts.append(jumps.sum())
    
    return sum(jump_counts)


def clip_gripper(df, max_val=252.0):
    gripper_cols = [c for c in df.columns if 'gripper' in c.lower()]

    for col in gripper_cols:
        sample = df[col].iloc[0]
        if isinstance(sample, np.ndarray):
            def clip_array(arr):
                if isinstance(arr, np.ndarray):
                    if arr.max() > max_val:
                        return np.clip(arr, None, max_val)
                return arr

            df[col] = df[col].apply(clip_array)

    return df


def process_task(task_path, output_path, task_type, task_name):
    print(f"\n{'='*70}")
    print(f"Processing: {task_name} ({task_type})")
    print(f"{'='*70}")

    output_path.mkdir(parents=True, exist_ok=True)
    episodes = sorted([d for d in task_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    total_eps = len(episodes)

    processed_eps = 0
    skipped_eps = []
    error_eps = []

    for ep_dir in episodes:
        ep_name = ep_dir.name
        parquet_file = ep_dir / 'timeseries.parquet'

        if not parquet_file.exists():
            skipped_eps.append((ep_name, "no parquet"))
            continue

        try:
            df = pd.read_parquet(parquet_file)
            jump_count = detect_jumps(df, task_type)

            if jump_count > 0:
                skipped_eps.append((ep_name, f"{jump_count} jumps"))
                print(f"  [x] {ep_name}: {jump_count} jumps")
                continue

            df = clip_gripper(df)
            
            out_ep_dir = output_path / ep_name
            out_ep_dir.mkdir(exist_ok=True)
            out_parquet = out_ep_dir / 'timeseries.parquet'

            # Write to tmp file first, then move
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            df.to_parquet(tmp_path, engine='pyarrow', compression='snappy')
            
            # Verify tmp file
            if tmp_path.stat().st_size == 0:
                raise Exception("tmp parquet file is empty")
            
            # Move to final location
            shutil.move(str(tmp_path), str(out_parquet))
            
            # Verify final file
            if out_parquet.stat().st_size == 0:
                raise Exception("final parquet file is empty")

            metadata_file = ep_dir / 'metadata.json'
            if metadata_file.exists():
                shutil.copy(metadata_file, out_ep_dir / 'metadata.json')

            processed_eps += 1
            print(f"  [+] {ep_name}: T={len(df)}")

        except Exception as e:
            error_eps.append((ep_name, str(e)))
            print(f"  [!] {ep_name}: {str(e)}")

    print(f"\nSummary for {task_name}:")
    print(f"  Total: {total_eps}, Processed: {processed_eps}, Skipped: {len(skipped_eps)}, Errors: {len(error_eps)}")

    return processed_eps, len(skipped_eps), len(error_eps), skipped_eps


def main():
    parser = argparse.ArgumentParser(description='Data repair')
    parser.add_argument('--input_dir', type=str,
                        default='/path/to/raw_franka_data')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/raw_franka_data_cleaned')

    args = parser.parse_args()
    base_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)

    print(f"Input: {base_dir}")
    print(f"Output: {output_base}")

    tasks = {
        'close_the_drawer_new': 'single',
        'pick_up_the_pen': 'single',
        'pick_up_the_steak': 'single',
        'pick_up_the_book': 'single',
        'pick_up_the_banana': 'dual',
        'pick_up_the_box': 'dual',
    }

    total_processed = 0
    total_skipped = 0
    total_errors = 0
    all_skipped = {}

    for task_name, task_type in tasks.items():
        task_path = base_dir / task_name
        output_path = output_base / task_name

        if not task_path.exists():
            continue

        processed, skipped, errors, skipped_list = process_task(
            task_path, output_path, task_type, task_name
        )
        total_processed += processed
        total_skipped += skipped
        total_errors += errors
        if skipped_list:
            all_skipped[task_name] = skipped_list

    skipped_file = output_base / 'skipped_episodes.txt'
    with open(skipped_file, 'w') as f:
        for task_name, skipped_list in all_skipped.items():
            f.write(f"\n{task_name}:\n")
            for ep_name, reason in skipped_list:
                f.write(f"  {ep_name}: {reason}\n")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Errors: {total_errors}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
