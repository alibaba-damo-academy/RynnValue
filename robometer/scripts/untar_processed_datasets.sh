#!/bin/bash
if [ -z "${ROBOMETER_PROCESSED_DATASETS_PATH:-$RBM_PROCESSED_DATASETS_PATH}" ]; then
    echo "ROBOMETER_PROCESSED_DATASETS_PATH (or RBM_PROCESSED_DATASETS_PATH) is not set"
    exit 1
fi

cd "${ROBOMETER_PROCESSED_DATASETS_PATH:-$RBM_PROCESSED_DATASETS_PATH}" || exit 1

# Track already processed archives to avoid duplicates
declare -A processed_archives

# First, handle split archives (.tar.partaa, .tar.partab, etc.)
echo "Processing split archives..."
for file in *.tar.partaa; do
    if [ -f "$file" ]; then
        # Get the base name without .partaa
        base_name="${file%.partaa}"
        
        echo "Extracting split archive: $base_name"
        # Concatenate all parts and extract
        cat "${base_name}.part"* | tar -xvf -

        # remove the parts if successfully extracted
        if [ $? -eq 0 ]; then
            rm "${base_name}.part"*
        else
            echo "Failed to extract $base_name, will need to retry and remove the failed parts"
            continue
        fi
        # Mark this base archive as processed
        processed_archives["$base_name"]=1

    fi
done

# Now handle split archives that look like .tar.part-aa, .tar.part-ab, etc.
echo "Processing split archives..."
for file in *.tar.part-aa; do
    if [ -f "$file" ]; then
        # Get the base name without .part-aa
        base_name="${file%.part-aa}"
        echo "Extracting split archive: $base_name"
        # Concatenate all parts and extract
        cat "${base_name}.part"* | tar -xvf -

        # remove the parts if successfully extracted
        if [ $? -eq 0 ]; then
            rm "${base_name}.part"*
        else
            echo "Failed to extract $base_name, will need to retry and remove the failed parts"
            continue
        fi
        # Mark this base archive as processed
        processed_archives["$base_name"]=1
    fi
done

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

# Now, some of the datasets are moved into a `processed_datasets` subdirectory, so move them out and delete 
# the overall processed_datasets directory if it exists.
if [ -d "processed_datasets" ]; then
    echo "Moving datasets out of processed_datasets subdirectory..."
    for dir in processed_datasets/*; do
        if [ -d "$dir" ]; then
            mv "$dir" .
        fi
    done
    rm -rf processed_datasets
    echo "Done moving datasets out of processed_datasets subdirectory!"
fi

# print which datasets might've failed
for file in *.tar; do
    if [ -z "${processed_archives[$file]}" ]; then
        echo "Failed to extract $file"
    fi
done
cd ..
echo "Done extracting all archives!"