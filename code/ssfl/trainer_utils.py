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


def prepare_training(model_name, global_model, num_clients, num_clusters=3):
    total_layer = get_total_layers(global_model)
    print(f"\nTotal layer in {model_name} is {total_layer}")
    arc_configs = np.linspace(1, total_layer - 1, num_clusters, dtype=int).tolist()
    clients_per_cluster = profile_resources(num_clients, num_clusters)
    return arc_configs, clients_per_cluster


def select_participating_clients(num_clients, frac):
    k = max(1, int(num_clients * frac))
    return random.sample(range(num_clients), k)


# --- THIS IS THE MODIFIED FUNCTION ---
def train_single_client(model_name, num_classes, arc_cfg, global_model,
                        device, in_channels, train_loader, val_loader, loss_fn,
                        local_epochs, cid,
                        hps: dict,
                        global_epoch): # MODIFIED: Changed 'lr' to 'hps: dict'
    """
    Trains a single client model using a given set of hyperparameters.
    """
    client_net, server_net, full_ref, _ = create_split_model(
        model_name, num_classes, arc_cfg,
        base_model=global_model, device=device, in_channels=in_channels
    )
    
    # --- Use client-specific hyperparameters from the 'hps' dictionary ---
    lr = hps.get('learning_rate', 0.001)
    weight_decay = hps.get('weight_decay', 5e-5)
    optimizer_name = hps.get('optimizer', 'AdamW')
    mu = hps.get('mu', 0.01) # Get the FedProx mu parameter

    # We create a detached copy so we can compare against it later.
    global_client_params = [param.detach().clone() for param in client_net.parameters()]

    # Select optimizer based on HPs
    if optimizer_name.lower() == 'sgd':
        opt_c = torch.optim.SGD(client_net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        opt_s = torch.optim.SGD(server_net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else: # Default to AdamW
        opt_c = torch.optim.AdamW(client_net.parameters(), lr=lr, weight_decay=weight_decay)
        opt_s = torch.optim.AdamW(server_net.parameters(), lr=lr, weight_decay=weight_decay)

    sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=local_epochs)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=local_epochs)

    client_net.train(); server_net.train()
    
    # Store results for the Analyzer Agent
    results = {'train_loss': [], 'test_acc': [], 'train_acc': []}

    for epoch in range(local_epochs):
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_c.zero_grad(); opt_s.zero_grad()

            client_feat = client_net(imgs)
            smashed = client_feat.detach().requires_grad_(True)
            out = server_net(smashed)
            loss = loss_fn(out, lbls)


            # --- 3. Add the FedProx proximal term to the client's loss ---
            prox_term = 0.0
            # Calculate the squared L2 norm between local and global model weights
            for local_param, global_param in zip(client_net.parameters(), global_client_params):
                prox_term += torch.pow(torch.norm(local_param - global_param), 2)
            
            # The total loss for the client includes the standard loss and the prox term
            fed_prox_loss = (mu / 2) * prox_term


            loss.backward()

            # Backpropagate the FedProx loss for the client model.
            # PyTorch automatically accumulates these gradients with any other gradients
            # that will be computed for the client network.
            fed_prox_loss.backward()



            client_feat.backward(smashed.grad)

            clip_grad_norm_(client_net.parameters(), 1.0)
            clip_grad_norm_(server_net.parameters(), 1.0)
            opt_c.step(); opt_s.step()
        sch_c.step(); sch_s.step()

        # Evaluate the client model at each local epoch for the analyzer
        temp_model = combine_client_server_models(client_net, server_net, full_ref.to(device), device, num_classes, arc_cfg)
        train_acc, train_loss = evaluate_model(temp_model, train_loader, device, loss_fn)
        test_acc, test_loss = evaluate_model(temp_model, val_loader, device, loss_fn)
        
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
        print(f"  Client {cid}, Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

    # Return weights, data size, and the results for the workflow
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