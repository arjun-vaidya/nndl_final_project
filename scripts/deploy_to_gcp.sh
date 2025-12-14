#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project)

echo "Packaging Project"
tar -czf project_bundle.tar.gz \
    --exclude='data/augmented_train_images' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='checkpoints' \
    --exclude='.ipynb_checkpoints' \
    src data requirements.txt README.md

echo "Uploading to GCP ($INSTANCE_NAME)"
gcloud compute scp project_bundle.tar.gz $INSTANCE_NAME:~ --zone=$ZONE
rm project_bundle.tar.gz