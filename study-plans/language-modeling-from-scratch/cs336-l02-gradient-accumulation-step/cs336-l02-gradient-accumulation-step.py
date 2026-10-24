import torch

def gradient_accumulation_step(
    param: torch.Tensor, microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor], lr: float,
) -> dict:
    """
    Returns new_param and full_grad tensors in a dictionary.
    """
    full_grad = torch.zeros_like(param)
    total_examples = 0

    for x, y in zip(microbatch_inputs, microbatch_targets):
        predictions = x @ param
        error = predictions - y

        grads = 2 * error.unsqueeze(1) * x

        full_grad += grads.sum(dim=0)
        total_examples += x.shape[0]

    full_grad /= total_examples

    new_param = param - lr * full_grad

    return {
        "new_param" : new_param,
        "full_grad" : full_grad
     }

        