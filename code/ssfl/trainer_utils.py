# code/ssfl/trainer_utils.py
import torch
from torch.nn.utils import clip_grad_norm_
from ssfl.aggregation import FedAvg, combine_client_server_models
from ssfl.model_splitter import create_split_model, get_total_layers, create_global_model
from ssfl.resource_profiler import profile_resources
from ssfl.utils import calculate_accuracy, save_model
import numpy as np
import random
from torch.utils.data import ConcatDataset, DataLoader
import copy

def create_optimizer(model_params, hps: dict):
    """Dynamically creates an optimizer based on hyperparameter suggestions."""
    optimizer_name = hps.get('optimizer', 'AdamW')
    lr = hps.get('learning_rate', 0.001)
    wd = hps.get('weight_decay', 0)
    momentum = hps.get('momentum', 0.9)

    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=momentum)
    # Add other optimizers here if needed
    else:  # Default to AdamW
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=wd)

def create_scheduler(optimizer, hps: dict, T_max: int):
    """Dynamically creates a learning rate scheduler."""
    scheduler_name = hps.get('scheduler', 'None')
    
    if scheduler_name.lower() == 'cosineannealinglr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name.lower() == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, T_max // 3), gamma=0.1)
    else:
        return None

def prepare_training(model_name, global_model, num_clients, num_clusters=3, fl_mode="splitfed"):
    """
    Prepares training configurations based on the FL mode.
    For 'splitfed', it sets up arc_configs and client-to-cluster mappings.
    For 'centralized', it simplifies to a single logical cluster.
    """
    if fl_mode == "splitfed":
        total_layer = get_total_layers(global_model)
        print(f"\nTotal layer in {model_name} is {total_layer}")
        # Ensure arc_configs are valid; total_layer - 1 is max split point for non-empty server
        arc_configs = np.linspace(1, max(1, total_layer - 1), num_clusters, dtype=int).tolist()
        clients_per_cluster = profile_resources(num_clients, num_clusters)
        return arc_configs, clients_per_cluster, total_layer
    elif fl_mode == "centralized":
        print("\n--- Centralized FL Mode: No model splitting for local training ---")
        # For centralized, we treat all clients as part of a single logical cluster.
        # arc_configs can be a dummy list, as it won't be used for splitting in train_single_client.
        arc_configs = [0] # A single dummy arc_config
        # All clients belong to cluster 0
        clients_per_cluster = {i: 0 for i in range(num_clients)}
        total_layer = get_total_layers(global_model) # Still relevant for model info
        return arc_configs, clients_per_cluster, total_layer
    else:
        raise ValueError(f"Unknown FL mode: {fl_mode}. Supported modes are 'splitfed' and 'centralized'.")


def prepare_training_old(model_name, global_model, num_clients, num_clusters=3):
    total_layer = get_total_layers(global_model)
    print(f"\nTotal layer in {model_name} is {total_layer}")
    arc_configs = np.linspace(1, total_layer - 1, num_clusters, dtype=int).tolist()
    clients_per_cluster = profile_resources(num_clients, num_clusters)
    return arc_configs, clients_per_cluster, total_layer


def select_participating_clients(num_clients, frac):
    k = max(1, int(num_clients * frac))
    return random.sample(range(num_clients), k)


def train_single_client(model_name, num_classes, arc_cfg, global_model,
                        device, in_channels, train_loader, val_loader, loss_fn,
                        cid, hps: dict, global_epoch: int, fl_mode: str): # Added fl_mode
    """
    Trains a single client model. Adapts based on 'splitfed' or 'centralized' mode.
    """
    results = {'train_loss': [], 'test_acc': [], 'train_acc': []}

    if fl_mode == "splitfed":
        dropout_rate = hps.get('client', {}).get('dropout_rate', 0.1) # Assuming this is for splitfed client
        client_net, server_net, full_ref, _ = create_split_model(
            model_name, num_classes, arc_cfg,
            base_model=global_model, device=device, in_channels=in_channels
        )

        client_hps = hps.get('client', {})
        server_hps = hps.get('server', {})
        mu = hps.get('mu', 0.01) # FedProx mu for split-FL client
        local_epochs = int(client_hps.get('local_epochs', 1))

        opt_c = create_optimizer(client_net.parameters(), client_hps)
        opt_s = create_optimizer(server_net.parameters(), server_hps)

        sch_c = create_scheduler(opt_c, client_hps, T_max=local_epochs)
        sch_s = create_scheduler(opt_s, server_hps, T_max=local_epochs)

        global_client_params = [param.detach().clone() for param in client_net.parameters()]

        client_net.train(); server_net.train()

        for epoch in range(local_epochs):
            for imgs, lbls in train_loader:
                if imgs.shape[0] <= 1:
                    print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                    continue
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt_c.zero_grad(); opt_s.zero_grad()
                client_feat = client_net(imgs)
                smashed = client_feat.detach().requires_grad_(True)
                out = server_net(smashed)

                loss = loss_fn(out, lbls)
                if mu > 0:
                    prox_term = 0.0
                    for local_param, global_param in zip(client_net.parameters(), global_client_params):
                        prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (mu / 2) * prox_term

                loss.backward()
                client_feat.backward(smashed.grad)
                clip_grad_norm_(client_net.parameters(), 1.0)
                clip_grad_norm_(server_net.parameters(), 1.0)
                opt_c.step(); opt_s.step()

            if sch_c: sch_c.step()
            if sch_s: sch_s.step()

        temp_model = combine_client_server_models(client_net, server_net, full_ref.to(device), device, num_classes, arc_cfg)
        train_acc, _ = evaluate_model(temp_model, train_loader, device, loss_fn)
        test_acc, _ = evaluate_model(temp_model, val_loader, device, loss_fn)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)

        print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        return client_net.state_dict(), server_net.state_dict(), len(train_loader.dataset), results

    elif fl_mode == "centralized":
        # Create a deep copy of the global model for local client training
        local_model = copy.deepcopy(global_model).to(device)

        # HPs for centralized mode (using client HPs from the agent suggestions)
        client_hps = hps.get('client', {})
        local_epochs = int(client_hps.get('local_epochs', 1))
        mu = hps.get('mu', 0.0) # FedProx mu for centralized FL, typically 0 but can be used

        optimizer = create_optimizer(local_model.parameters(), client_hps)
        scheduler = create_scheduler(optimizer, client_hps, T_max=local_epochs)

        
        # --- MODIFICATION START ---
        # Save initial global model parameters for proximal term calculation if mu > 0
        # IMPORTANT: Ensure these parameters are also on the same device as local_model
        global_model_params = [p.detach().clone().to(device) for p in global_model.parameters()]
        # --- MODIFICATION END ---

        local_model.train() # Set to training mode

        for epoch in range(local_epochs):
            for imgs, lbls in train_loader:
                if imgs.shape[0] <= 1:
                    print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                    continue
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()

                out = local_model(imgs)
                loss = loss_fn(out, lbls)

                if mu > 0:
                    prox_term = 0.0
                    for local_param, global_param in zip(local_model.parameters(), global_model_params):
                        prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (mu / 2) * prox_term

                loss.backward()
                clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

            if scheduler: scheduler.step()

        # Evaluate the locally trained full model
        train_acc, _ = evaluate_model(local_model, train_loader, device, loss_fn)
        test_acc, _ = evaluate_model(local_model, val_loader, device, loss_fn)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)

        print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        # Return full model state_dict in place of client_net.state_dict(), None for server_net
        return local_model.state_dict(), None, len(train_loader.dataset), results
    else:
        raise ValueError(f"Unsupported FL mode: {fl_mode}")

def train_single_client_old(model_name, num_classes, arc_cfg, global_model,
                        device, in_channels, train_loader, val_loader, loss_fn,
                        cid, hps: dict, global_epoch: int):
    """
    Trains a single client model using a fully dynamic set of hyperparameters
    for both the client and server models, as suggested by the HPO agent.
    """
    # Note: To use dropout_rate, your model definitions should accept it as a parameter
    dropout_rate = hps.get('client', {}).get('dropout_rate', 0.1)
    
    client_net, server_net, full_ref, _ = create_split_model(
        model_name, num_classes, arc_cfg,
        base_model=global_model, device=device, in_channels=in_channels
    )

    client_hps = hps.get('client', {})
    server_hps = hps.get('server', {})
    mu = hps.get('mu', 0.01)
    #local_epochs = client_hps.get('local_epochs', 1)
    local_epochs = int(client_hps.get('local_epochs', 1))

    opt_c = create_optimizer(client_net.parameters(), client_hps)
    opt_s = create_optimizer(server_net.parameters(), server_hps)
    
    sch_c = create_scheduler(opt_c, client_hps, T_max=local_epochs)
    sch_s = create_scheduler(opt_s, server_hps, T_max=local_epochs)
    
    global_client_params = [param.detach().clone() for param in client_net.parameters()]
    client_net.train(); server_net.train()
    results = {'train_loss': [], 'test_acc': [], 'train_acc': []}

    for epoch in range(local_epochs):
        for imgs, lbls in train_loader:
            # --- FINAL DEBUGGING CHECK AND SAFEGUARD ---
            # We will print the shape of every batch and skip any batch of size 1.
            # print(f"  --> Client {cid} training with batch of shape: {imgs.shape}")
            if imgs.shape[0] <= 1:
                print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                continue # Skip this iteration completely
            # --- END OF FIX ---
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_c.zero_grad(); opt_s.zero_grad()
            client_feat = client_net(imgs)
            smashed = client_feat.detach().requires_grad_(True)
            out = server_net(smashed)
            
            loss = loss_fn(out, lbls)
            if mu > 0:
                prox_term = 0.0
                for local_param, global_param in zip(client_net.parameters(), global_client_params):
                    prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                loss += (mu / 2) * prox_term
            
            loss.backward()
            client_feat.backward(smashed.grad)
            clip_grad_norm_(client_net.parameters(), 1.0)
            clip_grad_norm_(server_net.parameters(), 1.0)
            opt_c.step(); opt_s.step()

        if sch_c: sch_c.step()
        if sch_s: sch_s.step()

    temp_model = combine_client_server_models(client_net, server_net, full_ref.to(device), device, num_classes, arc_cfg)
    train_acc, _ = evaluate_model(temp_model, train_loader, device, loss_fn)
    test_acc, _ = evaluate_model(temp_model, val_loader, device, loss_fn)
    
    results['train_acc'].append(train_acc)
    results['test_acc'].append(test_acc)
    
    print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

    return client_net.state_dict(), server_net.state_dict(), len(train_loader.dataset), results


def cluster_fedavg(client_weights, server_weights, client_sizes):
    avg_client = FedAvg(client_weights, client_sizes)
    avg_server = FedAvg(server_weights, client_sizes)
    return avg_client, avg_server


def build_cluster_model(model_name, num_classes, arc_cfg,
                        global_model, device, in_channels,
                        client_weight, server_weight):
    fresh_c, fresh_s, full_ref, _ = create_split_model(
        model_name, num_classes, arc_cfg,
        base_model=global_model, device=device, in_channels=in_channels
    )
    fresh_c.load_state_dict(client_weight)
    fresh_s.load_state_dict(server_weight)

    cluster_model = combine_client_server_models(
        fresh_c, fresh_s, full_ref.to(device),
        device, num_classes, arc_cfg
    )
    return cluster_model


def evaluate_model(model, dataloader, device, loss_fn):
    model = model.to(device).eval()
    acc_sum, loss_sum, n = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss_sum += loss_fn(out, lbls).item() * imgs.size(0)
            acc_sum += calculate_accuracy(out, lbls) * imgs.size(0)
            n += imgs.size(0)

    # --- ADD THIS CHECK TO PREVENT ZeroDivisionError ---
    if n == 0:
        print(f"  WARNING: No samples processed during evaluation. Returning 0.0 for accuracy and loss.")
        return 0.0, 0.0 # Return 0 accuracy and 0 loss if no samples
    # --- END ADDITION ---
    return acc_sum / n, loss_sum / n


def global_fedavg(cluster_models, cluster_sizes):
    return FedAvg(cluster_models, cluster_sizes)

def _format_report(client_states: list) -> list:
    """Helper function to format the final YAML report for readability."""
    report_data = []
    for i, state in enumerate(client_states):
        client_report = {
            'client_id': i,
            'final_search_space': state.get('search_space', {}),
            'hpo_report': {}
        }
        # Correctly iterate over the hpo_report dictionary
        for epoch, report_entry in state.get('hpo_report', {}).items():
            report_key = f"Global Epoch: {epoch}"
            client_report['hpo_report'][report_key] = report_entry
        report_data.append(client_report)
    return report_data