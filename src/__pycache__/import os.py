import os
import torch
import torch.nn.utils.prune as prune

def get_model_size_kb(model):
    """ Returns the parameter memory footprint size in KB. """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024

def get_actual_size_kb(model):
    """ Simulates saving to disk to find actual KB threshold. """
    torch.save(model.state_dict(), "temp.p")
    size_kb = os.path.getsize("temp.p") / 1024
    os.remove("temp.p")
    return size_kb

def apply_pruning(model, amount=0.3):
    """ Post-training structural L1 norm pruning. """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

def apply_quantization(model):
    """ Applies Dynamic Quantization (Float32 to Int8). """
    model = model.cpu()
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def execute_green_ai_pipeline(model, prune_amount=0.3):
    """ Runs the Green AI Model Compression Stack. """
    pruned_model = apply_pruning(model, amount=prune_amount)
    final_model = apply_quantization(pruned_model)
    return final_model
