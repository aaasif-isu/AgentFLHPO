# code/ssfl/trainer.py
import torch
from ssfl.model_splitter import create_global_model
from ssfl.utils import ensure_dir, save_model
from ssfl.trainer_utils import (
    prepare_training,
    select_participating_clients,
    train_single_client,
    cluster_fedavg,
    build_cluster_model,
    evaluate_model,
    global_fedavg
)
from torch.utils.data import ConcatDataset, DataLoader

def train_model(model_name, num_classes, in_channels,
                train_loaders, val_loader,
                device, global_epochs, local_epochs,
                num_clients, imbalance_ratio, dataset_name, frac,
                train_subsets): # Added train_subsets for cluster and global train evaluation

    print(f"\nSplitFed: {global_epochs=}, {local_epochs=}, {num_clients=}, {frac=}")
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    global_model = create_global_model(model_name, num_classes, in_channels, device)

    best_global_accuracy, no_improvement_count = 0.0, 0
    patience, min_delta = 10, 0.01
    lr = 1e-3

    best_model_path = f"best_model/best_{dataset_name}_c{num_clients}_imb{imbalance_ratio}.pth"
    latest_model_path = f"latest_model/latest_{dataset_name}_c{num_clients}_imb{imbalance_ratio}.pth"
    ensure_dir("best_model"); ensure_dir("latest_model")

    arc_configs, clients_per_cluster = prepare_training(model_name, global_model, num_clients)

    for g_epoch in range(global_epochs):
        print(f"\n=== Global Epoch {g_epoch+1}/{global_epochs} ===")
        selected = select_participating_clients(num_clients, frac)

        cluster_state_dicts, cluster_sizes = [], []

        for c_id, arc_cfg in enumerate(arc_configs):
            members = [cid for cid in selected if clients_per_cluster[cid] == c_id]
            if len(members) < 1:
                continue

            print(f"Cluster {c_id} using arc_config={arc_cfg} with members {members}")

            local_client_w, local_server_w, local_sizes = [], [], []

            for cid in members:
                w_c, w_s, sz = train_single_client(
                    model_name, num_classes, arc_cfg, global_model,
                    device, in_channels, train_loaders[cid], val_loader, loss_fn,
                    lr, local_epochs, cid
                )
                local_client_w.append(w_c)
                local_server_w.append(w_s)
                local_sizes.append(sz)

            w_c, w_s = cluster_fedavg(local_client_w, local_server_w, local_sizes)
            cluster_model = build_cluster_model(
                model_name, num_classes, arc_cfg,
                global_model, device, in_channels, w_c, w_s
            )

            # Evaluate cluster model
            cluster_train_subset = ConcatDataset([train_subsets[i] for i in members])
            cluster_train_loader = DataLoader(cluster_train_subset, batch_size=128, shuffle=True)
            acc_train, loss_train = evaluate_model(cluster_model, cluster_train_loader, device, loss_fn)
            acc_test, loss_test = evaluate_model(cluster_model, val_loader, device, loss_fn)
            print(f"  Cluster {c_id} Train Acc {acc_train:.2f}%, Test Acc {acc_test:.2f}%")


            cluster_state_dicts.append(cluster_model.state_dict())
            cluster_sizes.append(sum(local_sizes))

        if cluster_state_dicts:
            global_weights = global_fedavg(cluster_state_dicts, cluster_sizes)
            global_model.load_state_dict(global_weights)
            global_model = global_model.to(device)
            save_model(global_model, latest_model_path)

            # Evaluate global model
            all_train_subset = ConcatDataset(train_subsets)
            all_train_loader = DataLoader(all_train_subset, batch_size=128, shuffle=False)
            g_train_acc, g_train_loss = evaluate_model(global_model, all_train_loader, device, loss_fn)
            g_test_acc, g_test_loss = evaluate_model(global_model, val_loader, device, loss_fn)
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
    return global_model