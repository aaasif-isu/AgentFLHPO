# trainer_utils.py
import torch
from torch.nn.utils import clip_grad_norm_
from ssfl.aggregation import FedAvg, combine_client_server_models
from ssfl.model_splitter import create_split_model, get_total_layers
from ssfl.resource_profiler import profile_resources
from ssfl.utils import calculate_accuracy, save_model
import numpy as np
import random


def prepare_training(model_name, global_model, num_clients, num_clusters=3):
    total_layer = get_total_layers(global_model)
    print(f"\nTotal layer in {model_name} is {total_layer}")
    arc_configs = np.linspace(1, total_layer - 1, num_clusters, dtype=int).tolist()
    clients_per_cluster = profile_resources(num_clients, num_clusters)
    return arc_configs, clients_per_cluster


def select_participating_clients(num_clients, frac):
    k = max(1, int(num_clients * frac))
    return random.sample(range(num_clients), k)


def train_single_client(model_name, num_classes, arc_cfg, global_model,
                        device, in_channels, train_loader, loss_fn,
                        lr, local_epochs):
    client_net, server_net, _, _ = create_split_model(
        model_name, num_classes, arc_cfg,
        base_model=global_model, device=device, in_channels=in_channels
    )
    opt_c = torch.optim.AdamW(client_net.parameters(), lr, weight_decay=5e-5)
    opt_s = torch.optim.AdamW(server_net.parameters(), lr, weight_decay=5e-5)
    sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=local_epochs)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=local_epochs)

    client_net.train(); server_net.train()
    for _ in range(local_epochs):
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_c.zero_grad(); opt_s.zero_grad()

            client_feat = client_net(imgs)
            smashed = client_feat.detach().requires_grad_(True)
            out = server_net(smashed)
            loss = loss_fn(out, lbls)

            loss.backward()
            client_feat.backward(smashed.grad)

            clip_grad_norm_(client_net.parameters(), 1.0)
            clip_grad_norm_(server_net.parameters(), 1.0)
            opt_c.step(); opt_s.step()
        sch_c.step(); sch_s.step()

    return client_net.state_dict(), server_net.state_dict(), len(train_loader.dataset)


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
