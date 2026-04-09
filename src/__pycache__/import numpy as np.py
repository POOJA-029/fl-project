import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def calculate_demographic_parity(y_pred, protected_attr):
    """ Demographic Parity Difference (prediction rate equality). """
    group_0_preds = y_pred[protected_attr == 0]
    group_1_preds = y_pred[protected_attr == 1]
    
    if len(group_0_preds) == 0 or len(group_1_preds) == 0:
        return 0.0
        
    rate_0 = np.mean(group_0_preds)
    rate_1 = np.mean(group_1_preds)
    
    return abs(rate_0 - rate_1)

def calculate_equal_opportunity(y_true, y_pred, protected_attr):
    """ Equal Opportunity Difference (TPR equality). """
    group_0_mask = (protected_attr == 0) & (y_true == 1)
    group_1_mask = (protected_attr == 1) & (y_true == 1)
    
    tpr_0 = np.mean(y_pred[group_0_mask]) if np.sum(group_0_mask) > 0 else 0
    tpr_1 = np.mean(y_pred[group_1_mask]) if np.sum(group_1_mask) > 0 else 0
    
    return abs(tpr_0 - tpr_1)

def apply_reweighing(X, y, protected_attr):
    """ Data Resampling Simulation for Mitigation (Oversampling minority). """
    df = np.column_stack((X, y, protected_attr))
    
    group_counts = {}
    groups = {}
    for g_y in [0, 1]:
        for g_p in [0, 1]:
            mask = (y == g_y) & (protected_attr == g_p)
            elements = df[mask]
            groups[(g_y, g_p)] = elements
            group_counts[(g_y, g_p)] = len(elements)
            
    max_count = max(group_counts.values()) if len(group_counts) > 0 else 1
    
    resampled_df = []
    for g_y in [0, 1]:
        for g_p in [0, 1]:
            elements = groups[(g_y, g_p)]
            if len(elements) > 0:
                repeat_times = max_count // len(elements)
                remainder = max_count % len(elements)
                upsampled = np.vstack([elements] * repeat_times + [elements[:remainder]])
                resampled_df.append(upsampled)
                
    if not resampled_df:
        return X, y, protected_attr
        
    resampled_df = np.vstack(resampled_df)
    np.random.shuffle(resampled_df)
    
    X_new = resampled_df[:, :-2]
    y_new = resampled_df[:, -2]
    p_new = resampled_df[:, -1]
    
    return X_new, y_new, p_new

def get_fair_dataloader(X, y, protected_attr, batch_size=16):
    """ Returns a DataLoader reweighed for Fairness. """
    X_fair, y_fair, p_fair = apply_reweighing(X, y, protected_attr)
    dataset = TensorDataset(torch.tensor(X_fair, dtype=torch.float32), 
                            torch.tensor(y_fair, dtype=torch.float32).unsqueeze(1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
