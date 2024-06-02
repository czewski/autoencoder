#!/bin/bash

# List of Python files to run
scripts=(
    "ae_reconstruct_runner.py"
    #"ae_target_runner.py"
    "ae_token_reconstruct_runner.py"
    "dae_reconstruct_runner.py"
    #"dae_target_runner.py"
    "dae_token_reconstruct_runner.py"
    "vae_reconstruct_runner.py"
    #"vae_target_runner.py"
    "vae_token_reconstruct_runner.py"
)

# Loop through the list and run each Python file
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python3 "$script"
    if [ $? -ne 0 ]; then
        echo "Error occurred while running $script. Exiting."
        exit 1
    fi
done

echo "All scripts executed successfully."