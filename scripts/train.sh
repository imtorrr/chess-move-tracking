#!/bin/bash
# For local use.
# Make sure to install dependencies from pyproject.toml
# e.g. `uv sync`

# Create data directory if it doesn't exist
mkdir -p data

# Download and extract data
gdown https://drive.google.com/uc?id=1a6p9fm7RNY2X86PFnBsSLml_D4JJkR1h -O data/merged.tar.gz
tar -xvzf data/merged.tar.gz -C data/

# Configure yolo
yolo settings tensorboard=True

# Run training
yolo detect train data=data/merged/data.yaml model=yolo11l.pt epochs=100 imgsz=640 degrees=15.0 flipud=0.1 fliplr=0.1 cache=True batch=-1 patience=25
