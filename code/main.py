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
        frac=config["frac"]
    )

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Training completed.")

if __name__ == "__main__":
    main()