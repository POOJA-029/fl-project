import os
import torch
import torch.nn.utils.prune as prune

def get_model_size_kb(model):
    """
    Returns the size of the model in kilobytes.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_kb = (param_size + buffer_size) / 1024
    return size_all_kb

def get_actual_size_kb(model):
    """
    Simulates saving the state_dict to disk and measuring its size in KB.
    """
    torch.save(model.state_dict(), "temp.p")
    size_kb = os.path.getsize("temp.p") / 1024
    os.remove("temp.p")
    return size_kb


def apply_pruning(model, amount=0.3):
    """
    Applies unstructured L1 pruning to the Linear layers.
    Removes the specified percentage of the connections with the lowest L1-norm.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # Prune `amount`% of the connections
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make the pruning permanent
            prune.remove(module, 'weight')
    return model

def apply_quantization(model):
    """
    Applies Post-Training Dynamic Quantization (PTQ) to the Linear layers.
    Converts 32-bit floating point weights to 8-bit integers.
    """
    # Move model to CPU explicitly because quantization is typically done for CPU inference
    model = model.cpu()
    
    # Needs to be in eval mode for quantization
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def execute_green_ai_pipeline(model, prune_amount=0.3):
    """
    Returns the optimized model (Quantized + Pruned).
    """
    # 1. Prune
    pruned_model = apply_pruning(model, amount=prune_amount)
    
    # 2. Quantize
    final_model = apply_quantization(pruned_model)
    
    return final_model
