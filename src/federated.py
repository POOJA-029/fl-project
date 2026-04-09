import torch
import copy
from .model import train_model

def get_model_parameters(model):
    """
    Extracts the parameters of the given PyTorch model.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """
    Sets the parameters of the given PyTorch model.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = dict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def federated_averaging(global_model, client_models, client_weights):
    """
    Performs Federated Averaging (FedAvg).
    Aggregates the weights from client models based on their data size (client_weights).
    """
    global_dict = global_model.state_dict()
    
    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    for k in global_dict.keys():
        # Weighted average of each layer
        global_dict[k] = sum(
            client_models[i].state_dict()[k] * normalized_weights[i] 
            for i in range(len(client_models))
        )
        
    global_model.load_state_dict(global_dict)
    return global_model

def simulate_fl_round(global_model, client_dataloaders, client_epochs=2, lr=0.01):
    """
    Simulates one round of Federated Learning.
    Each client trains on their local data and returns the updated model.
    """
    client_models = []
    client_weights = []
    
    for dataloader in client_dataloaders:
        # Clone global model for local training
        local_model = copy.deepcopy(global_model)
        
        # Train locally
        train_model(local_model, dataloader, epochs=client_epochs, lr=lr)
        
        client_models.append(local_model)
        client_weights.append(len(dataloader.dataset)) # Weight by number of samples
        
    # Aggregate back to global model
    global_model = federated_averaging(global_model, client_models, client_weights)
    return global_model
