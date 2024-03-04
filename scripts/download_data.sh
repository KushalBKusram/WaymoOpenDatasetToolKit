#!/bin/bash

# Function to display usage information
function usage {
    echo "Usage: $0 bucket_name destination_folder [-m]"
    echo "Options:"
    echo "  -m    Enable parallel (multi-threaded/multi-processing) operations"
    exit 1
}

# Check if the number of arguments is valid
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    usage
fi

# Parse command-line arguments
bucket_name="$1"
destination_folder="$2"
parallel_flag="$3"

# Check if parallel flag is provided and set the gsutil command accordingly
if [ "$parallel_flag" == "-m" ]; then
    gsutil_cmd="gsutil -m cp -r gs://$bucket_name/* $destination_folder"
else
    gsutil_cmd="gsutil cp -r gs://$bucket_name/* $destination_folder"
fi

# Execute the gsutil command
echo "Downloading contents of bucket $bucket_name to $destination_folder..."
eval $gsutil_cmd
