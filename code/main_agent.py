import os
import sys
import torch
from agents.workflow import run_workflow
from ssfl.utils import load_dataset, subsample_dataset, partition_data_non_iid_random, create_dataloaders, Tee # Ensure all these utilities are in ssfl.utils
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

    train_subsets = partition_data_non_iid_random(
        train_subset, config["num_clients"], config["imbalance_ratio"], config["min_samples_per_client"]
    )
    # Default batch_size is now managed by HP agent, but used here for initial DataLoader creation
    train_loaders = create_dataloaders(train_subsets, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_log_dir = f"SFL_{config['dataset_name']}_{config['imbalance_ratio']}"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # The log filename now uses num_clusters from the config, not local_epochs directly
    log_filename = os.path.join(base_log_dir, f"{config['num_clients']}_clients_{config['num_clusters']}_clusters.txt")
    try:
        sys.stdout = Tee(log_filename)
    except Exception as e:
        print(f"Failed to redirect stdout: {e}")

    print(f"Training with {config['num_clients']} clients, Dataset: {config['dataset_name']}")
    print(f"Non-IIDness: {config['imbalance_ratio']}, Model: {config['model_name']}")
    # Removed print for 'local_epochs' as it's now dynamically controlled by HP agent per client.
    # The initial value is 2 local epochs per client training step, but agents can adjust the search space.

    print(f"Number of Clusters: {config['num_clusters']}") # Add this for clarity

    # Run LLM agent workflow
    state = run_workflow(
        num_clients=config["num_clients"],
        model_name=config["model_name"], # Changed 'model' to 'model_name'
        dataset=config["dataset_name"],
        train_loaders=train_loaders,
        val_loader=test_loader,
        device=device,
        num_classes=num_classes,
        in_channels=in_channels,
        imbalance_ratio=config["imbalance_ratio"],
        num_clusters=config["num_clusters"]
    )

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Training completed.")
    print(f"Best global accuracy achieved: {state['best_global_accuracy']:.2f}%") # Print final best accuracy

if __name__ == "__main__":
    main()