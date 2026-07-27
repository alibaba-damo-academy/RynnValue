#!/bin/bash
if [ -z "${ROBOMETER_PROCESSED_DATASETS_PATH:-$RBM_PROCESSED_DATASETS_PATH}" ]; then
    echo "ROBOMETER_PROCESSED_DATASETS_PATH (or RBM_PROCESSED_DATASETS_PATH) is not set"
    exit 1
fi

cd "${ROBOMETER_PROCESSED_DATASETS_PATH:-$RBM_PROCESSED_DATASETS_PATH}" || exit 1

# Track already processed archives to avoid duplicates
declare -A processed_archives

# Then, handle regular tar files (skip those that were split archives)
echo "Processing regular tar files..."
for file in *.tar; do
    if [ -f "$file" ]; then
        # Skip if this was already processed as a split archive
        if [ -z "${processed_archives[$file]}" ]; then
            echo "Extracting: $file"
            tar -xvf "$file"

            # remove the tar file only if it was successfully extracted
            if [ $? -eq 0 ]; then
                processed_archives["$file"]=1
                rm "$file"
            else
                echo "Failed to extract $file, will need to retry and remove the failed tar file"
                continue
            fi
        fi
    fi
done

# print which datasets might've failed
for file in *.tar; do
    if [ -z "${processed_archives[$file]}" ]; then
        echo "Failed to extract $file"
    fi
done
cd ..
echo "Done extracting all archives!"