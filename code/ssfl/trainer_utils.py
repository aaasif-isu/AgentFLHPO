# code/ssfl/trainer_utils.py
import torch
from torch.nn.utils import clip_grad_norm_
from ssfl.aggregation import FedAvg, combine_client_server_models
from ssfl.model_splitter import create_split_model, get_total_layers
from ssfl.resource_profiler import profile_resources
from ssfl.utils import calculate_accuracy, save_model
import numpy as np
import random
from torch.utils.data import ConcatDataset, DataLoader

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

def prepare_training(model_name, global_model, num_clients, num_clusters=3):
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