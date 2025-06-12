import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import sys
import os
from torchvision import datasets, transforms


import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split

from torch.utils.data import Subset, random_split 



random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def save_model(model, path):
    torch.save(model.state_dict(), path)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.0 * correct.float() / preds.shape[0]
    return acc.item()

def create_model(model_name: str, num_classes: int, in_channels=3):
    from torchvision import models
    from ssfl.model_splitter import CNNBase
    if model_name.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name.lower() == 'cnn':
        return CNNBase(num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def create_model_old(model_name: str, num_classes: int):
    from torchvision import models
    if model_name.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['dx'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.image_dir}/{self.data.loc[idx, 'image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.data.loc[idx, 'dx']]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()

    # Set default image size
    if image_size is None:
        if dataset_name == "cifar10":
            image_size = 32
        elif dataset_name == "mnist" or dataset_name == "femnist":
            image_size = 28
        elif dataset_name == "imagenet" or dataset_name == "ham10000":
            image_size = 224
        else:
            raise ValueError("Provide image_size for unknown dataset")

    #in_channels = 1 if dataset_name in ["mnist", "femnist"] else 3
    # Set native in_channels for dataset
    native_in_channels = 1 if dataset_name in ["mnist", "femnist"] else 3

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        DatasetClass = datasets.CIFAR10
        train_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform)
        test_dataset = DatasetClass(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        in_channels = 3  # ResNet18 and CNN both use 3 channels for CIFAR-10

    elif dataset_name == "mnist":
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if native_in_channels == 1 else x),  # Repeat for 3-channel models
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if in_channels == 3 else x),  # Use in_channels
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if model_name.lower() == "resnet18" else x),  # Conditional transform
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # convert grayscale to RGB
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        DatasetClass = datasets.MNIST
        train_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform)
        test_dataset = DatasetClass(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        in_channels = 3  # Always return 3 channels for MNIST to support ResNet18

    elif dataset_name == "femnist":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if native_in_channels == 1 else x),  # Repeat for 3-channel models
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if in_channels == 3 else x),  # Use in_channels
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if model_name.lower() == "resnet18" else x),  # Conditional transform
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if in_channels == 3 else x),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = FEMNISTDataset(data_path="./data/FEMNIST/train", transform=transform)
        test_dataset = FEMNISTDataset(data_path="./data/FEMNIST/test", transform=transform)
        num_classes = 62
        in_channels = 3  # Always return 3 channels for FEMNIST to support ResNet18

    elif dataset_name == "ham10000":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7630, 0.5456, 0.5702],
                                 std=[0.1409, 0.1523, 0.1695])  # HAM10000 stats or ImageNet
        ])
        full_dataset = HAM10000Dataset(
            csv_file='./data/HAM10000/HAM10000_metadata.csv',
            image_dir='./data/HAM10000/images',
            transform=transform
        )
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        num_classes = 7

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset, num_classes, image_size, in_channels



def load_dataset_old(dataset_name, transform):
    if dataset_name.lower() == "cifar10":
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")
    return train_dataset, test_dataset

def subsample_dataset(dataset, sample_fraction):
    num_samples = int(len(dataset) * sample_fraction)
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices)

def partition_data_non_iid_random(subset, num_clients, imbalance_factor, min_samples_per_client):
    full_dataset = subset.dataset
    subset_indices = subset.indices
    num_classes = 10  # CIFAR-10
    subset_targets = np.array(full_dataset.targets)[subset_indices]
    class_indices = {i: np.where(subset_targets == i)[0] for i in range(num_classes)}
    client_data_indices = {i: [] for i in range(num_clients)}

    for client_id in range(num_clients):
        client_class_distribution = np.random.dirichlet(np.ones(num_classes) * imbalance_factor)
        for cls in range(num_classes):
            num_samples = int(client_class_distribution[cls] * len(class_indices[cls]))
            if num_samples > 0:
                client_data_indices[client_id].extend(np.random.choice(class_indices[cls], num_samples, replace=False))
        np.random.shuffle(client_data_indices[client_id])
        if len(client_data_indices[client_id]) < min_samples_per_client:
            print(f"Warning: Client {client_id} has {len(client_data_indices[client_id])} samples.")
            remaining_samples_needed = min_samples_per_client - len(client_data_indices[client_id])
            available_indices = np.concatenate([class_indices[cls] for cls in range(num_classes)])
            additional_indices = np.random.choice(available_indices, remaining_samples_needed, replace=False)
            client_data_indices[client_id].extend(additional_indices)

    client_data_indices = {client_id: subset_indices[indices] for client_id, indices in client_data_indices.items()}
    return [Subset(subset.dataset, indices) for indices in client_data_indices.values()]

def create_dataloaders(subsets, batch_size, shuffle):
    return [DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=True) for subset in subsets]


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

def split_client_data(client_subset, val_split_ratio=0.2):
    """
    Splits a client's dataset (a PyTorch Subset) into training and validation.
    """
    num_total = len(client_subset)
    if num_total == 0: # Handle empty client subset
        return client_subset, Subset(client_subset.dataset, [])

    num_val = int(num_total * val_split_ratio)
    if num_val == 0 and num_total > 1 and val_split_ratio > 0: num_val = 1 # Ensure at least 1 val sample
    
    num_train = num_total - num_val
    if num_train <= 0: # If not enough data for a split
        return client_subset, Subset(client_subset.dataset, [])

    train_local_indices, val_local_indices = random_split(range(num_total), [num_train, num_val])
    
    original_dataset_train_indices = [client_subset.indices[i] for i in train_local_indices.indices]
    original_dataset_val_indices = [client_subset.indices[i] for i in val_local_indices.indices]

    return Subset(client_subset.dataset, original_dataset_train_indices), \
           Subset(client_subset.dataset, original_dataset_val_indices)

def prepare_client_dataloaders_for_hpo(
    client_data_subsets, # Output of partition_data_non_iid_random
    batch_size,
    val_split_ratio=0.2,
    num_workers=0, # Make num_workers configurable
    pin_memory=False # Make pin_memory configurable
):
    """
    Prepares client-specific training and validation DataLoaders for HPO.

    Args:
        client_data_subsets (list): List of PyTorch Subsets, one for each client.
        batch_size (int): The batch size for the DataLoaders.
        val_split_ratio (float): Fraction of a client's data for local validation.
        num_workers (int): Number of worker processes for DataLoader.
        pin_memory (bool): If True, DataLoader will copy Tensors into CUDA pinned memory.

    Returns:
        tuple: (client_train_loaders, client_val_loaders)
               client_train_loaders: List of DataLoaders for client training.
               client_val_loaders: Dict (client_id -> DataLoader) for client validation.
    """
    client_train_loaders = []
    client_val_loaders = {} 

    for i, per_client_subset in enumerate(client_data_subsets):
        if len(per_client_subset) == 0: # Handle clients with no data
            empty_ds = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
            client_train_loaders.append(DataLoader(empty_ds, batch_size=batch_size))
            client_val_loaders[i] = DataLoader(empty_ds, batch_size=batch_size)
            continue
        
        client_local_train_data, client_local_val_data = split_client_data(
            per_client_subset, 
            val_split_ratio=val_split_ratio
        )

        train_loader = DataLoader(
            client_local_train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        client_train_loaders.append(train_loader)

        # Ensure validation loader is created even if client_local_val_data is empty after split
        if len(client_local_val_data) > 0:
            val_loader = DataLoader(
                client_local_val_data, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )
        else: # Create an empty DataLoader if validation set is empty
            empty_ds_val = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
            val_loader = DataLoader(empty_ds_val, batch_size=batch_size)
            
        client_val_loaders[i] = val_loader
            
    return client_train_loaders, client_val_loaders
            
class Tee(object):
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log_file = open(filename, mode)
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()

    