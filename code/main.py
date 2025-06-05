# code/main.py
import os
import sys
import torch

from ssfl.trainer import train_model
from ssfl.utils import *
import yaml

def load_config(path="model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Load raw datasets
    train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(config["dataset_name"])
    print(f"Dataset: {config['dataset_name'].upper()}")
    print(f"Number of Classes: {num_classes}")
    print(f"Image Size: {image_size} x {image_size}")

    train_subset = subsample_dataset(train_dataset, config["train_sample_fraction"])
    test_subset = subsample_dataset(test_dataset, config["test_sample_fraction"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_log_dir = f"SFL_{config['dataset_name']}_{config['imbalance_ratio']}"
    os.makedirs(base_log_dir, exist_ok=True)
    log_filename = os.path.join(base_log_dir, f"{config['num_clients']}_clients_{config['local_epochs']}_local.txt")
    try:
        sys.stdout = Tee(log_filename)
    except Exception as e:
        print(f"Failed to redirect stdout: {e}")

    train_subsets = partition_data_non_iid_random(
        train_subset, config["num_clients"], config["imbalance_ratio"], config["min_samples_per_client"]
    )
    train_loaders = create_dataloaders(train_subsets, batch_size=config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False)

    # No need to create model_ofm - trainer will handle model creation
    print(f"Training with {config['num_clients']} clients, {config['local_epochs']} local epochs")
    print(f"Non-IIDness: {config['imbalance_ratio']}, Dataset: {config['dataset_name']}")

    train_model(
        model_name=config["model_name"],  # Pass model_name instead of model_ofm
        num_classes=num_classes,
        in_channels=in_channels,
        train_loaders=train_loaders,
        val_loader=test_loader,
        device=device,
        global_epochs=config["global_epochs"],
        local_epochs=config["local_epochs"],
        num_clients=config["num_clients"],
        imbalance_ratio=config["imbalance_ratio"],
        dataset_name=config["dataset_name"],
        frac=config["frac"],
        train_subsets=train_subsets # Pass train_subsets
    )

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Training completed.")

if __name__ == "__main__":
    main()



# # code/main.py
# import os
# import sys
# import torch
# import yaml
# from torch.utils.data import DataLoader

# from ssfl.trainer import train_model
# from ssfl.utils import (
#     load_dataset,
#     subsample_dataset,
#     partition_data_non_iid_random,
#     Tee,
#     split_client_data # You still need this function in utils.py
# )

# def load_config(path="model_config.yaml"):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def main():
#     # === Setup (Same as your original) ===
#     config = load_config()
#     train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(config["dataset_name"])
#     train_subset = subsample_dataset(train_dataset, config["train_sample_fraction"])
#     test_subset = subsample_dataset(test_dataset, config["test_sample_fraction"])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Logging setup
#     base_log_dir = f"SFL_{config['dataset_name']}_{config['imbalance_ratio']}"
#     os.makedirs(base_log_dir, exist_ok=True)
#     log_filename = os.path.join(base_log_dir, f"{config['num_clients']}_clients_{config['local_epochs']}_local.txt")
#     sys.stdout = Tee(log_filename) # Assuming Tee is defined in your utils

#     # === Data Preparation (with minimal change for HPO) ===

#     # 1. Partition data among clients (Your existing logic)
#     client_data_subsets = partition_data_non_iid_random(
#         train_subset, 
#         config["num_clients"], 
#         config["imbalance_ratio"], 
#         config["min_samples_per_client"]
#     )

#     # 2. Create Train and Validation loaders for each client
#     client_train_loaders = []
#     client_val_loaders = {}
#     for i, client_subset in enumerate(client_data_subsets):
#         # For each client, split its data into local train and val sets
#         train_data, val_data = split_client_data(client_subset, val_split_ratio=0.2)
#         client_train_loaders.append(DataLoader(train_data, batch_size=config["batch_size"], shuffle=True))
#         client_val_loaders[i] = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)

#     # 3. Create global test loader (Your existing logic)
#     test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False)

#     # === HPO Parameters ===
#     initial_lr = config.get("learning_rate", 0.001)
#     fixed_local_epochs = config.get("local_epochs", 2)
#     # This path is still needed for the trainer to load the HPO config
#     hp_config_path = os.path.join(os.path.dirname(__file__), "agent", "hp_config.yaml")

#     print(f"Starting training with HPO. Fixed local epochs per turn: {fixed_local_epochs}")

#     # === Call train_model ONCE (with updated arguments for HPO) ===
#     # The signature of train_model in trainer.py must accept these arguments
#     train_model(
#         model_name=config["model_name"],
#         num_classes=num_classes,
#         in_channels=in_channels,
#         train_loaders=client_train_loaders,
#         val_loader=test_loader,
#         client_val_loaders=client_val_loaders, # Pass the local validation loaders
#         device=device,
#         global_epochs=config["global_epochs"],
#         fixed_local_epochs_per_turn=fixed_local_epochs, # Pass the fixed local epochs
#         num_clients=config["num_clients"],
#         imbalance_ratio=config["imbalance_ratio"],
#         dataset_name=config["dataset_name"],
#         frac=config["frac"],
#         initial_lr=initial_lr, # Pass the initial learning rate
#         hp_config_path=hp_config_path # Pass the path to HP config
#     )

#     if isinstance(sys.stdout, Tee):
#         sys.stdout.close()
#         sys.stdout = sys.__stdout__
#     print("Training completed.")

# if __name__ == "__main__":
#     main()