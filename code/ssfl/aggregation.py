import torch
import copy
from ssfl.model_splitter import ResNetBase, CNNBase
from torchvision.models.resnet import ResNet  # Add this import
from collections import OrderedDict
import torch.nn as nn


def FedAvg(weight_dicts, sizes=None):
    """Equal or weighted average of state-dicts."""
    if not weight_dicts:
        raise ValueError("empty FedAvg list")
    if sizes is None:
        sizes = [1] * len(weight_dicts)
    total = sum(sizes)
    w_avg = copy.deepcopy(weight_dicts[0])
    for k in w_avg.keys():
        if w_avg[k].dtype.is_floating_point:
            w_avg[k] *= sizes[0]
            for i in range(1, len(weight_dicts)):
                w_avg[k] += weight_dicts[i][k] * sizes[i]
            w_avg[k] /= total
        # int / bool tensors: keep first client’s copy
    return w_avg

def combine_client_server_models(client_submodel: nn.Module,
                                 server_submodel: nn.Module,
                                 full_template:  nn.Module,
                                 device: str,
                                 num_classes: int,
                                 arc_config: int):
    """
    Stitch a client half and server half back into a *single* full network.

    Parameters
    ----------
    client_submodel : the trained client slice
    server_submodel : the trained server slice
    full_template   : an un-trained copy of the complete model
                      (CNNBase / ResNetBase / VGGBase) that provides the key
                      layout for .state_dict()
    device          : "cpu" or "cuda:0"
    num_classes     : final classifier size (not used, kept for signature compat)
    arc_config      : split point (needed only for sanity prints)

    Returns
    -------
    nn.Module   —  a full model with merged weights, living on `device`
    """
    # 1. start from the template’s state-dict
    merged_sd = full_template.state_dict()

    # 2. update with client weights (keys already match: layers 0 … arc_config-1)
    merged_sd.update(client_submodel.state_dict())

    # 3. update with server weights
    server_sd = server_submodel.state_dict()

    #    – if your server kept the name "classifier", rename → "fc" here
    # for k in list(server_sd.keys()):
    #     if k.startswith("classifier"):
    #         server_sd[k.replace("classifier", "fc", 1)] = server_sd.pop(k)

    merged_sd.update(server_sd)

    # 4. load back into a *fresh* copy of the template
    full_model = copy.deepcopy(full_template).to(device)
    full_model.load_state_dict(merged_sd)

    return full_model