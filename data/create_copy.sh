#!/bin/bash

# Check if the correct number of arguments is provided
if [ -z "$1" ] || [ -z "$2" ]; then
	  echo "Usage: $0 <number_of_copies> <full_path_to_file>"
	    exit 1
fi

# Number of copies to create
N=$1

# Full path to the original file
original_file="$2"

# Extract the directory and base filename from the full path
file_dir=$(dirname "$original_file")
file_name=$(basename "$original_file")

# Remove all extensions from the filename
base_name="${file_name%%.*}"

# Extract all extensions (e.g., .json.gz)
extensions="${file_name#*.}"

# Check if the original file exists
if [ ! -f "$original_file" ]; then
	  echo "Error: $original_file does not exist."
	    exit 1
fi

# Loop to create N copies
for (( idx=1; idx<=N; idx++ ))
do
	  new_file="${file_dir}/${base_name}_${idx}.${extensions}"
	    cp "$original_file" "$new_file"
	      echo "Created file: $new_file"
      done

      echo "Finished creating $N copies of $original_file."
