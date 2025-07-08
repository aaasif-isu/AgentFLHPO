# code/main.py
import yaml
import torch
import os
import sys
import logging
import threading # <-- 1. IMPORT THREADING

# --- Your existing imports ---
from ssfl.utils import (
    load_dataset,
    partition_data_non_iid_random,
    partition_text_non_iid_dirichlet,
    subsample_dataset,
    Tee
)
from ssfl.trainer import train_model

# --- Imports for the new parallel workflow ---
from agent.cpu_worker import background_cpu_work # <-- 2. IMPORT THE CPU WORKER
from agent.shared_state import results_queue     # <-- 3. IMPORT THE SHARED QUEUE

def load_config(path="model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # --- Logging Setup (remains the same) ---
    log_dir = config.get("log_dir", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{config.get('num_clients', 'N')}_clients_run.log")
    try:
        sys.stdout = Tee(log_filename)
    except Exception as e:
        print(f"Failed to redirect stdout: {e}")

    # --- Data Loading and Partitioning (remains the same) ---
    train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(config["dataset_name"])
    print(f"Dataset: {config['dataset_name'].upper()}")
    train_subset = subsample_dataset(train_dataset, config["train_sample_fraction"])
    test_subset = subsample_dataset(test_dataset, config["test_sample_fraction"])

    if config["dataset_name"].lower() == 'shakespeare_d':
        # This assumes a new function, 'get_data_by_client', which loads the
        # naturally partitioned data from all_data.json
        all_client_data = get_data_by_client(dataset_path=config["data_path"], dataset_name='shakespeare')
        
        train_subsets = []
        # The number of clients is now determined by the data itself
        num_clients = len(all_client_data.keys()) 

        for client_id in all_client_data.keys():
            # You would need to create a Dataset or Subset from the client's data
            # 'all_client_data[client_id]' contains the grouped data for one actor
            client_dataset = create_subset_from_grouped_data(all_client_data[client_id])
            train_subsets.append(client_dataset)

        # You no longer need to use 'imbalance_factor' or 'min_samples_per_client' 
        # for this specific partitioning method.

        # You may still want to check if the total number of clients matches 1129
        # as mentioned in the paper.
        print(f"Loaded {num_clients} clients from the dataset's inherent partitioning.")

    elif config["dataset_name"].lower() == 'shakespeare':
        train_subsets = partition_text_non_iid_dirichlet(
            dataset=train_subset,
            num_clients=config["num_clients"],
            imbalance_factor=config["imbalance_ratio"], # You can reuse this config
            min_samples_per_client=config["min_samples_per_client"] 
        )
    else:
        train_subsets = partition_data_non_iid_random(
            train_subset, config["num_clients"], config["imbalance_ratio"], config["min_samples_per_client"]
            )

    val_batch_size = config.get('val_batch_size', 128)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=val_batch_size, shuffle=False, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 4. PREPARE SHARED STATE FOR PARALLEL WORK ---
    # This logic is moved here from trainer.py to be shared between threads.
    hp_config_path = os.path.join("agent", "hp_config.yaml")
    with open(hp_config_path, 'r') as f:
        initial_search_space = yaml.safe_load(f)

    num_clients = config['num_clients']
    client_states = [{"search_space": initial_search_space.copy(), "concrete_hps": None, "hpo_report": {}, "last_analysis": None} for _ in range(num_clients)]

    # --- 5. START BACKGROUND CPU WORKER ---
    print("Starting background CPU worker...")
    # NOTE: The agents are instantiated inside your workflow, so we only pass client_states.
    worker_thread = threading.Thread(
        target=background_cpu_work,
        args=(client_states,)
    )
    worker_thread.daemon = True # This allows the main program to exit even if the thread is running.
    worker_thread.start()

    print(f"Training with {num_clients} clients.")
    print(f"Non-IIDness: {config['imbalance_ratio']}, Dataset: {config['dataset_name']}")

    # --- 6. MODIFIED CALL TO train_model ---
    # This now includes the client_states argument at the end.
    train_model(
        model_name=config["model_name"],
        num_classes=num_classes,
        in_channels=in_channels,
        train_subsets=train_subsets,
        val_loader=test_loader,
        device=device,
        global_epochs=config["global_epochs"],
        num_clients=num_clients,
        imbalance_ratio=config["imbalance_ratio"],
        dataset_name=config["dataset_name"],
        frac=config["frac"],
        config=config,
        client_states=client_states # <-- Pass the shared state object
    )

    # --- 7. GRACEFUL SHUTDOWN ---
    print("\nTraining complete. Waiting for final analysis tasks to finish...")
    results_queue.put((None, None))  # Send the "stop" signal to the worker.
    results_queue.join()  # Wait for the queue to be fully processed before exiting.
    print(" All tasks complete. System shutting down.")


    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Training completed.")

if __name__ == "__main__":
    main()






# # code/main.py
# import yaml
# import torch
# import os
# import sys
# import logging

# # Assuming these functions exist in your ssfl.utils module based on your original file
# from ssfl.utils import (
#     load_dataset,
#     partition_data_non_iid_random,
#     partition_text_non_iid_dirichlet,
#     subsample_dataset, # From your original main_2
#     Tee              # From your original main_2
# )
# from ssfl.trainer import train_model

# # --- User-provided helper function, placed directly in main.py ---
# def load_config(path="model_config.yaml"):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def main():
#     config = load_config()

#     # NOTE: The call to a `set_seed` function has been removed as it does not exist in your project.
#     # For reproducible research, you may want to implement a seed-setting utility in the future.

#     # --- Logging Setup ---
#     log_dir = config.get("log_dir", "logs")
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
    
#     # Corrected log filename that avoids dynamic HPs
#     log_filename = os.path.join(log_dir, f"{config.get('num_clients', 'N')}_clients_run.log")

#     try:
#         # Using the Tee class from your original code to log stdout
#         sys.stdout = Tee(log_filename)
#     except Exception as e:
#         print(f"Failed to redirect stdout: {e}")

#     # --- Data Loading and Partitioning (from your original file) ---
#     train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(config["dataset_name"])
#     print(f"Dataset: {config['dataset_name'].upper()}")
#     print(f"Number of Classes: {num_classes}")
#     print(f"Image Size: {image_size} x {image_size}")

#     # Using subsampling logic from your original file
#     train_subset = subsample_dataset(train_dataset, config["train_sample_fraction"])
#     test_subset = subsample_dataset(test_dataset, config["test_sample_fraction"])

    
#     if config["dataset_name"].lower() == 'shakespeare':
#         print("Using non-IID Dirichlet partitioning for Shakespeare dataset.")
#         # Use our new, correct function for text data
#         train_subsets = partition_text_non_iid_dirichlet(
#             dataset=train_subset,
#             num_clients=config["num_clients"],
#             imbalance_factor=config["imbalance_ratio"], # You can reuse this config
#             min_samples_per_client=config["min_samples_per_client"] 
#         )
#     else:
#         print("Using non-IID random-class partitioning for image dataset.")
#         # Keep the original function for your image classification datasets
#         train_subsets = partition_data_non_iid_random(
#                 train_subset, config["num_clients"], config["imbalance_ratio"], config["min_samples_per_client"]
#             )





    

#     # --- ADD THIS DEBUGGING BLOCK ---
#     # print("\n" + "="*40)
#     # print("--- Verifying Client Dataset Sizes ---")
#     # all_ok = True
#     # for i, subset in enumerate(train_subsets):
#     #     if len(subset) <= 1:
#     #         print(f"  - WARNING: Client {i} has only {len(subset)} sample(s). This WILL cause a BatchNorm error.")
#     #         all_ok = False
#     #     else:
#     #         print(f"  - Client {i} has {len(subset)} samples. (OK)")
#     # if all_ok:
#     #     print("--- All client datasets are large enough. ---")
#     # print("="*40 + "\n")
#     # --- END DEBUGGING BLOCK ---

#     # --- DataLoader Creation (Corrected Logic) ---
#     # The training DataLoaders are no longer created here.
#     # Only the validation/test loader is created, using a specific config parameter.
#     val_batch_size = config.get('val_batch_size', 128) # Uses a safe default of 128
#     test_loader = torch.utils.data.DataLoader(test_subset, batch_size=val_batch_size, shuffle=False, drop_last=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"Training with {config['num_clients']} clients.")
#     print(f"Non-IIDness: {config['imbalance_ratio']}, Dataset: {config['dataset_name']}")

#     # --- Final, Corrected call to train_model ---
#     # This call matches the refactored train_model signature, passing subsets instead of loaders.
#     train_model(
#         model_name=config["model_name"],
#         num_classes=num_classes,
#         in_channels=in_channels,
#         train_subsets=train_subsets,
#         val_loader=test_loader,
#         device=device,
#         global_epochs=config["global_epochs"],
#         num_clients=config["num_clients"],
#         imbalance_ratio=config["imbalance_ratio"],
#         dataset_name=config["dataset_name"],
#         frac=config["frac"],
#         config=config
#     )

#     sys.stdout.close()
#     sys.stdout = sys.__stdout__
#     print("Training completed.")

# if __name__ == "__main__":
#     main()