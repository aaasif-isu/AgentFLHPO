# code/main.py
import yaml
import torch
import os
import sys
import logging

# Assuming these functions exist in your ssfl.utils module based on your original file
from ssfl.utils import (
    load_dataset,
    partition_data_non_iid_random,
    partition_text_non_iid_dirichlet,
    subsample_dataset, # From your original main_2
    Tee              # From your original main_2
)
from ssfl.trainer import train_model

# --- User-provided helper function, placed directly in main.py ---
def load_config(path="model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # NOTE: The call to a `set_seed` function has been removed as it does not exist in your project.
    # For reproducible research, you may want to implement a seed-setting utility in the future.

    # --- Logging Setup ---
    log_dir = config.get("log_dir", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Corrected log filename that avoids dynamic HPs
    log_filename = os.path.join(log_dir, f"{config.get('num_clients', 'N')}_clients_run.log")

    try:
        # Using the Tee class from your original code to log stdout
        sys.stdout = Tee(log_filename)
    except Exception as e:
        print(f"Failed to redirect stdout: {e}")

    # --- Data Loading and Partitioning (from your original file) ---
    train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(config["dataset_name"])
    print(f"Dataset: {config['dataset_name'].upper()}")
    print(f"Number of Classes: {num_classes}")
    print(f"Image Size: {image_size} x {image_size}")

    # Using subsampling logic from your original file
    train_subset = subsample_dataset(train_dataset, config["train_sample_fraction"])
    test_subset = subsample_dataset(test_dataset, config["test_sample_fraction"])

    
    if config["dataset_name"].lower() == 'shakespeare':
        print("Using non-IID Dirichlet partitioning for Shakespeare dataset.")
        # Use our new, correct function for text data
        train_subsets = partition_text_non_iid_dirichlet(
            dataset=train_subset,
            num_clients=config["num_clients"],
            imbalance_factor=config["imbalance_ratio"], # You can reuse this config
            min_samples_per_client=config["min_samples_per_client"] 
        )
    else:
        print("Using non-IID random-class partitioning for image dataset.")
        # Keep the original function for your image classification datasets
        train_subsets = partition_data_non_iid_random(
                train_subset, config["num_clients"], config["imbalance_ratio"], config["min_samples_per_client"]
            )





    

    # --- ADD THIS DEBUGGING BLOCK ---
    # print("\n" + "="*40)
    # print("--- Verifying Client Dataset Sizes ---")
    # all_ok = True
    # for i, subset in enumerate(train_subsets):
    #     if len(subset) <= 1:
    #         print(f"  - WARNING: Client {i} has only {len(subset)} sample(s). This WILL cause a BatchNorm error.")
    #         all_ok = False
    #     else:
    #         print(f"  - Client {i} has {len(subset)} samples. (OK)")
    # if all_ok:
    #     print("--- All client datasets are large enough. ---")
    # print("="*40 + "\n")
    # --- END DEBUGGING BLOCK ---

    # --- DataLoader Creation (Corrected Logic) ---
    # The training DataLoaders are no longer created here.
    # Only the validation/test loader is created, using a specific config parameter.
    val_batch_size = config.get('val_batch_size', 128) # Uses a safe default of 128
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=val_batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training with {config['num_clients']} clients.")
    print(f"Non-IIDness: {config['imbalance_ratio']}, Dataset: {config['dataset_name']}")

    # --- Final, Corrected call to train_model ---
    # This call matches the refactored train_model signature, passing subsets instead of loaders.
    train_model(
        model_name=config["model_name"],
        num_classes=num_classes,
        in_channels=in_channels,
        train_subsets=train_subsets,
        val_loader=test_loader,
        device=device,
        global_epochs=config["global_epochs"],
        num_clients=config["num_clients"],
        imbalance_ratio=config["imbalance_ratio"],
        dataset_name=config["dataset_name"],
        frac=config["frac"],
        config=config
    )

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Training completed.")

if __name__ == "__main__":
    main()