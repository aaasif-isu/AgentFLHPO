# # code/ssfl/trainer.py

import torch
import yaml
import os
from torch.utils.data import ConcatDataset, DataLoader

# Original imports
from ssfl.model_splitter import create_global_model
from ssfl.utils import ensure_dir, save_model
from ssfl.trainer_utils import (
    prepare_training, select_participating_clients, build_cluster_model, 
    evaluate_model, _format_report
)
from ssfl.trainer_utils import cluster_fedavg, global_fedavg

# --- 1. Import the new strategy classes ---
# Make sure your strategies.py file is in the ssfl folder or adjust path.
from ssfl.strategies import AgentStrategy, FixedStrategy, RandomSearchStrategy, BO_Strategy , SHA_Strategy

def train_model(model_name, num_classes, in_channels,
                train_loaders, val_loader,
                device, global_epochs, local_epochs,
                num_clients, imbalance_ratio, dataset_name, frac,
                train_subsets, config): # Pass the whole config dictionary

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    global_model = create_global_model(model_name, num_classes, in_channels, device)

    hp_config_path = os.path.join(os.path.dirname(__file__), "..", "agent", "hp_config.yaml")
    with open(hp_config_path, 'r') as f:
        initial_search_space = yaml.safe_load(f)
    
    # --- 2. HPO STRATEGY SELECTION ---
    hpo_config = config.get('hpo_strategy', {})
    strategy_name = hpo_config.get('method', 'fixed') # Default to 'fixed' if not specified
    
    # Initialize with 'hpo_report' dictionary instead of 'history' list
    #client_states = [{"search_space": initial_search_space.copy(), "hpo_report": {}} for i in range(num_clients)]
    # --- MODIFICATION 7: Add "last_analysis" key to the state dictionary ---
    client_states = [{"search_space": initial_search_space.copy(), "hpo_report": {}, "last_analysis": None} for i in range(num_clients)]

    
    hpo_strategy = None

    # Common arguments for all strategies
    strategy_args = {
        "initial_search_space": initial_search_space,
        "client_states": client_states,
        "num_clients": num_clients
    }

    if strategy_name == 'agent':
        hpo_strategy = AgentStrategy(**strategy_args)
    elif strategy_name == 'random_search':
        hpo_strategy = RandomSearchStrategy(**strategy_args)
    elif strategy_name == 'sha':
        # You can pass SHA-specific configs from your YAML file here if needed
        hpo_strategy = SHA_Strategy(**strategy_args, population_size=27, elimination_rate=3)
    elif strategy_name == 'bo':
        hpo_strategy = BO_Strategy(**strategy_args)
    else: # Default to the fixed baseline strategy
        strategy_args['fixed_hps'] = hpo_config.get('fixed_hps')
        hpo_strategy = FixedStrategy(**strategy_args)  

    print(f"--- Using HPO Strategy: {strategy_name.upper()} ---")

    # --- Setup code from original trainer (Unchanged) ---
    best_global_accuracy, no_improvement_count = 0.0, 0
    patience, min_delta = 10, 0.01
    best_model_path = f"best_model/best_{dataset_name}_c{num_clients}_imb{imbalance_ratio}.pth"
    latest_model_path = f"latest_model/latest_{dataset_name}_c{num_clients}_imb{imbalance_ratio}.pth"
    ensure_dir("best_model"); ensure_dir("latest_model")
    arc_configs, clients_per_cluster = prepare_training(model_name, global_model, num_clients)

    # --- Main Training Loop ---
    for g_epoch in range(global_epochs):
        print(f"\n=== Global Epoch {g_epoch+1}/{global_epochs} ===")
        selected = select_participating_clients(num_clients, frac)
        cluster_state_dicts, cluster_sizes = [], []

        for c_id, arc_cfg in enumerate(arc_configs):
            members = [cid for cid in selected if clients_per_cluster[cid] == c_id]
            if not members:
                continue

            print(f"Cluster {c_id} using arc_config={arc_cfg} with members {members}")
            local_client_w, local_server_w, local_sizes = [], [], []

            # --- NEW: Initialize a list to store peer history for this cluster/epoch ---
            cluster_peer_history = []

            for cid in members:
                # --- 3. DELEGATE TO THE STRATEGY ---
                context = {
                    "client_id": cid,
                    "cluster_id": c_id,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "peer_history": cluster_peer_history, # Pass the history of previous peers
                    "training_args": {
                        "model_name": model_name, "num_classes": num_classes, "arc_cfg": arc_cfg,
                        "global_model": global_model, "device": device, "in_channels": in_channels,
                        "train_loader": train_loaders[cid], "val_loader": val_loader, "loss_fn": loss_fn,
                        "local_epochs": local_epochs,
                        "global_epoch": g_epoch

                    }
                }

                # The strategy object handles all HPO and training logic
                hps, w_c, w_s, sz, final_state = hpo_strategy.get_hyperparameters(context)
            
                
                # Append results for aggregation
                local_client_w.append(w_c)
                local_server_w.append(w_s)
                local_sizes.append(sz)

                # Call the updated state method, now passing the global epoch
                if final_state is not None and isinstance(hpo_strategy, AgentStrategy):
                    hpo_strategy.update_persistent_state(cid, context, final_state)

                    # --- NEW: Create a summary of the completed run for the next peer ---
                    if final_state.get('last_analysis'):
                        key_insight = final_state.get('last_analysis', {}).get('decision_summary', 'Analysis failed.')
                        peer_summary = {
                            "client_id": cid,
                            "hps_used": final_state.get('hps', {}),
                            "result_and_decision": f"Achieved {final_state.get('results', {}).get('test_acc', [0.0])[-1]:.2f}% Acc. Analyzer Decision: '{key_insight}'"
                        }
                        cluster_peer_history.append(peer_summary)
            

            # --- Aggregation and Evaluation Logic (Unchanged) ---
            if not local_client_w:
                continue
            
            w_c, w_s = cluster_fedavg(local_client_w, local_server_w, local_sizes)
            cluster_model = build_cluster_model(model_name, num_classes, arc_cfg, global_model, device, in_channels, w_c, w_s)
            
            cluster_train_subset = ConcatDataset([train_subsets[i] for i in members])
            cluster_train_loader = DataLoader(cluster_train_subset, batch_size=128, shuffle=True)
            acc_train, _ = evaluate_model(cluster_model, cluster_train_loader, device, loss_fn)
            acc_test, _ = evaluate_model(cluster_model, val_loader, device, loss_fn)
            print(f"  Cluster {c_id} Train Acc {acc_train:.2f}%, Test Acc {acc_test:.2f}%")

            cluster_state_dicts.append(cluster_model.state_dict())
            cluster_sizes.append(sum(local_sizes))

        if cluster_state_dicts:
            global_weights = global_fedavg(cluster_state_dicts, cluster_sizes)
            global_model.load_state_dict(global_weights)
            
            all_train_subset = ConcatDataset(train_subsets)
            all_train_loader = DataLoader(all_train_subset, batch_size=128, shuffle=False)
            g_train_acc, _ = evaluate_model(global_model, all_train_loader, device, loss_fn)
            g_test_acc, _ = evaluate_model(global_model, val_loader, device, loss_fn)
            print(f"Global Epoch {g_epoch+1}: Train Acc {g_train_acc:.2f}%, Test Acc {g_test_acc:.2f}%")

            if g_test_acc > best_global_accuracy + min_delta:
                best_global_accuracy, no_improvement_count = g_test_acc, 0
                save_model(global_model, best_model_path)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping.")
                    break

    print(f"Best Global Accuracy: {best_global_accuracy:.2f}%")

    # --- Final Save of HPO States ---
    # Get the relative path from your config file
    relative_path = config.get('hpo_checkpoint_path', 'agent/client_hpo_states.yaml')

    # --- THIS IS THE SIMPLIFIED FIX ---
    # Construct the correct path by going up one directory ('..') from the current script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_states_path = os.path.join(current_dir, '..', relative_path)
    
    print(f"\n--- Saving final HPO states for all clients to {final_states_path} ---")
    try:
        final_states_to_save = _format_report(hpo_strategy.client_states)

        # Add the client_id to each state for clarity before saving
        for i, state in enumerate(final_states_to_save):
            state['client_id'] = i
            # Rename search_space to final_search_space for clarity in the final report
            if 'search_space' in state:
                state['final_search_space'] = state.pop('search_space')


        checkpoint_dir = os.path.dirname(final_states_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        with open(final_states_path, 'w') as f:
            import copy
            final_report_to_dump = copy.deepcopy(final_states_to_save)

            yaml.dump(final_report_to_dump, f, indent=4, default_flow_style=False, sort_keys=False)

            # yaml.dump(final_states_to_save, f, indent=4, sort_keys=False, default_flow_style=False)
        print("--- Final states saved successfully. ---")

    except Exception as e:
        print(f"Error: Could not save final HPO states. {e}")
    return global_model