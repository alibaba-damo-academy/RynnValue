#!/usr/bin/env python3
"""
Simple validation script for the Robometer dataset format.
Checks fields and data types only.
"""

import argparse
from typing import Any

import numpy as np

from datasets import Dataset, load_from_disk


def validate_dataset_fields_and_types(dataset: Dataset, sample_size: int = 10) -> dict[str, Any]:
    """Validate dataset fields and data types."""

    print(f"Validating dataset fields and data types on {sample_size} sample entries...")

    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {"dataset_size": len(dataset), "samples_checked": 0},
    }

    # Expected schema for the new format
    expected_fields = [
        "id",
        "task",
        "lang_vector",
        "data_source",
        "frames",
        "is_robot",
        "quality_label",
        "preference_group_id",
        "preference_rank",
    ]

    # Check if dataset has features
    if not hasattr(dataset, "features") or dataset.features is None:
        validation_results["valid"] = False
        validation_results["errors"].append("Dataset has no features defined")
        return validation_results

    print(f"Dataset size: {len(dataset)} entries")
    print(f"Dataset features: {list(dataset.features.keys())}")

    # Check required fields
    for field_name in expected_fields:
        if field_name not in dataset.features:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Missing required field: {field_name}")
        else:
            print(f"✓ Field '{field_name}' present")

    # Sample entries for validation
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    validation_results["stats"]["samples_checked"] = len(sample_indices)

    for idx in sample_indices:
        trajectory = dataset[idx]

        try:
            # Validate each field
            if not isinstance(trajectory["id"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'id' is not a string")

            if not isinstance(trajectory["task"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'task' is not a string")

            # lang_vector should be length-384 sequence
            lv = trajectory["lang_vector"]
            if isinstance(lv, np.ndarray):
                if lv.shape != (384,):
                    validation_results["errors"].append(
                        f"Trajectory {idx}: 'lang_vector' shape is {lv.shape}, expected (384,)"
                    )
            elif isinstance(lv, list):
                if len(lv) != 384:
                    validation_results["errors"].append(
                        f"Trajectory {idx}: 'lang_vector' length is {len(lv)}, expected 384"
                    )
                else:
                    # check element types
                    if not all(isinstance(x, (int, float, np.floating, np.integer)) for x in lv):
                        validation_results["warnings"].append(
                            f"Trajectory {idx}: 'lang_vector' contains non-numeric elements"
                        )
            else:
                validation_results["errors"].append(f"Trajectory {idx}: 'lang_vector' has unexpected type {type(lv)}")

            if not isinstance(trajectory["data_source"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'data_source' is not a string")

            if not isinstance(trajectory["frames"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'frames' is not a string path")

            if not isinstance(trajectory["is_robot"], bool):
                validation_results["errors"].append(f"Trajectory {idx}: 'is_robot' is not a boolean")

            if not isinstance(trajectory["quality_label"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'quality_label' is not a string")
            else:
                if trajectory["quality_label"] not in {"successful", "failure", "suboptimal"}:
                    validation_results["warnings"].append(
                        f"Trajectory {idx}: 'quality_label' has unexpected value '{trajectory['quality_label']}'"
                    )

            # preference fields can be None
            if trajectory.get("preference_group_id") is not None and not isinstance(
                trajectory["preference_group_id"], str
            ):
                validation_results["errors"].append(
                    f"Trajectory {idx}: 'preference_group_id' is neither None nor string"
                )
            if trajectory.get("preference_rank") is not None and not isinstance(trajectory["preference_rank"], int):
                validation_results["errors"].append(f"Trajectory {idx}: 'preference_rank' is neither None nor int")

            # Print sample task for first trajectory
            if idx == sample_indices[0]:
                print("\nSample task from first trajectory:")
                print(f"  Task: {trajectory['task']}")
                print(f"  ID: {trajectory['id']}")

        except Exception as e:
            validation_results["errors"].append(f"Trajectory {idx}: Error during validation: {e}")

    if validation_results["errors"]:
        validation_results["valid"] = False

    return validation_results


def print_validation_summary(validation_results: dict[str, Any]):
    """Print validation summary."""

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    status = "✅ PASS" if validation_results["valid"] else "❌ FAIL"
    print(f"Status: {status}")

    print(f"Dataset size: {validation_results['stats']['dataset_size']}")
    print(f"Samples checked: {validation_results['stats']['samples_checked']}")

    if validation_results.get("errors"):
        print(f"\nErrors ({len(validation_results['errors'])}):")
        for error in validation_results["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(validation_results["errors"]) > 10:
            print(f"  ... and {len(validation_results['errors']) - 10} more errors")

    if validation_results.get("warnings"):
        print(f"\nWarnings ({len(validation_results['warnings'])}):")
        for warning in validation_results["warnings"][:5]:  # Show first 5 warnings
            print(f"  - {warning}")
        if len(validation_results["warnings"]) > 5:
            print(f"  ... and {len(validation_results['warnings']) - 5} more warnings")

    print("=" * 50)


def main():
    """Main validation function."""

    parser = argparse.ArgumentParser(description="Validate dataset fields and data types")
    parser.add_argument("dataset_path", help="Path to the HuggingFace dataset")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of samples to check")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Dataset loaded successfully.")

    # Run validation
    validation_results = validate_dataset_fields_and_types(dataset, args.sample_size)

    # Print summary
    print_validation_summary(validation_results)

    # Exit with error code if validation failed
    if not validation_results["valid"]:
        exit(1)


if __name__ == "__main__":
    main()
