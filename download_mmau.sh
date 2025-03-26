#!/bin/bash

set -e  # Exit immediately if a command fails

mkdir -p data && cd data

# Clone the repository if it does not exist
if [ ! -d "MMAU" ]; then
    git clone https://github.com/Sakshi113/MMAU.git
fi

cd MMAU

# Download from Google Drive
FILEID="1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA"
FILENAME="test-mini-audios.tar.gz"

echo "Downloading ${FILENAME} from Google Drive..."

if [ -f "$FILENAME" ]; then
    echo " -> File already exists. Skipping download."
else 
    wget --quiet --show-progress \
        --no-check-certificate \
        --no-clobber \
        "https://drive.usercontent.google.com/download?id=${FILEID}&confirm=t" \
        -O "${FILENAME}"
fi

echo ""
echo "Extracting ${FILENAME}..."
tar --checkpoint=2500 --checkpoint-action=dot -xzf "$FILENAME"
echo ""
echo "✅ Completed extracting!"
echo ""

# Ask if we want to remove the compressed file
read -p " -> Do you want to delete the compressed file? [Y/n] " -n 1 -r
echo    # move to a new line after read input

if [[ $REPLY =~ ^[Yy]$ || -z $REPLY ]]; then
    echo "Removing ${FILENAME}..."
    rm "$FILENAME"
else
    echo "Keeping ${FILENAME}"
fi
